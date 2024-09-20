import librosa
import warnings
import soundfile
import torch
import yaml
import time
import rfwave
import reflow
import re
import kaldiio
import torchaudio
import torch.cuda.amp as amp

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm


ENABLE_FP16 = False
COMPILE = False
torch.set_float32_matmul_precision('high')


def load_config(config_yaml):
    with open(config_yaml, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def create_instance(config):
    for k, v in config['init_args'].items():
        if isinstance(v, dict) and 'class_path' in v and 'init_args' in v:
            config['init_args'][k] = create_instance(v)
    return eval(config['class_path'])(**config['init_args'])


def load_model(model_dir, device, last=False):
    config_yaml = Path(model_dir) / 'config.yaml'
    if last:
        ckpt_fp = list(Path(model_dir).rglob("last.ckpt"))
        if len(ckpt_fp) == 0:
            raise ValueError(f"No checkpoint found in {model_dir}")
        elif len(ckpt_fp) > 1:
            warnings.warn(f"More than 1 checkpoints found in {model_dir}")
            ckpt_fp = sorted([fp for fp in ckpt_fp], key=lambda x: x.stat().st_ctime)[-1:]
        ckpt_fp = ckpt_fp[0]
        print(f'using last ckpt form {str(ckpt_fp)}')
    else:
        ckpt_fp = [fp for fp in list(Path(model_dir).rglob("*.ckpt")) if 'last' not in fp.stem]
        ckpt_fp = sorted(ckpt_fp, key=lambda x: int(re.search('_step=(\d+)_', x.stem).group(1)))[-1]
        print(f'using best ckpt form {str(ckpt_fp)}')

    config = load_config(config_yaml)
    exp = create_instance(config['model'])

    model_dict = torch.load(ckpt_fp, map_location='cpu')
    exp.load_state_dict(model_dict['state_dict'])

    if COMPILE:
        exp.reflow.backbone = torch.compile(exp.reflow.backbone)

    exp.eval()
    exp.to(device)
    return exp

# test code for rf2 model.
def sample_teacher(teacher_model, mel, encodec_bandwidth_id=None):
    def remove_image(pred):
        ss = pred.shape
        fs = (ss[0] // teacher_model.num_bands, ss[1] * teacher_model.num_bands, ss[2])
        pred = teacher_model.place_joint_subband(pred.reshape(fs))
        pred = teacher_model.stft(teacher_model.istft(pred))
        pred = teacher_model.get_joint_subband(pred).reshape(ss)
        return pred

    with torch.no_grad():
        bandwidth_id = torch.tile(torch.arange(teacher_model.num_bands, device=mel.device), (mel.size(0),))
        z0 = teacher_model.get_joint_z0(mel)
        t0 = torch.zeros((z0.size(0),), device=mel.device)
        mel = mel.repeat_interleave(teacher_model.num_bands, 0)
        pred = teacher_model.get_pred(z0, t0, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
        pred = remove_image(pred)
        t_ = torch.rand((mel.shape[0] // teacher_model.num_bands,), device=mel.device) * 0.6 + 0.2     # 0.2 ~ 0.8
        t_ = torch.repeat_interleave(t_, teacher_model.num_bands, dim=0)
        step_t = torch.einsum('b,bij->bij', t_, pred)
        x_t_psuedo = z0 + step_t
        pred_teacher = teacher_model.get_pred(
            x_t_psuedo, t_, mel, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id)
        # pred_teacher = remove_image(pred_teacher)
        step_1_t = torch.einsum('b,bij->bij', 1 - t_, pred_teacher)
        pred_teacher = step_t + step_1_t
        z1 = z0 + pred_teacher
    z1 = teacher_model.place_joint_subband(z1.reshape(z1.size(0) // teacher_model.num_bands, -1, z1.size(2)))
    wave = teacher_model.get_wave(z1)
    return wave


def copy_synthesis(exp, y, N=1000):
    features = exp.feature_extractor(y)
    start = time.time()
    sample = exp.reflow.sample_ode(features, N=N)[-1]
    # samples = sample_teacher(exp.reflow, features)
    cost = time.time() - start
    l = min(sample.size(-1), y.size(-1))
    rvm_loss = exp.rvm(sample[..., :l], y[..., :l])
    recon = sample.detach().cpu().numpy()[0]
    return recon, cost, rvm_loss


def copy_synthesis_encodec(exp, y, N=1000):
    num_encodec_bandwidths = len(exp.feature_extractor.bandwidths)
    recons = {}
    costs = {}
    rmv_losses = {}
    for encodec_bandwidth_id in range(num_encodec_bandwidths):
        # encodec_bandwidth_id is set in feature_extractor.forward
        features = exp.feature_extractor(y, encodec_bandwidth_id=encodec_bandwidth_id)
        encodec_audio = exp.feature_extractor.encodec(y[None, :])
        encodec_audio = encodec_audio.detach().cpu().numpy()[0, 0]
        start = time.time()
        encodec_bandwidth_id = torch.tensor([encodec_bandwidth_id], dtype=torch.long, device=y.device)
        sample = exp.reflow.sample_ode(features, encodec_bandwidth_id=encodec_bandwidth_id, N=N)[-1]
        cost = time.time() - start
        l = min(sample.size(-1), y.size(-1))
        rvm_loss = exp.rvm(sample[..., :l], y[..., :l])
        recon = sample.detach().cpu().numpy()[0]
        recons[f'bw_{encodec_bandwidth_id.item()}'] = recon
        recons[f'enc_bw_{encodec_bandwidth_id.item()}'] = encodec_audio
        costs[f'bw_{encodec_bandwidth_id.item()}'] = cost
        rmv_losses[f'bw_{encodec_bandwidth_id.item()}'] = rvm_loss
    return recons, costs, rmv_losses


def voc(model_dir, wav_dir, save_dir, guidance_scale, N=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(model_dir, device=device, last=True)
    if exp.reflow.guidance_scale == 1. and guidance_scale is not None and guidance_scale > 1.:
        warnings.warn("The original does not use classifier-free guidance. cfg argument is omitted")
    if guidance_scale is not None:
        print(f"Original guidance_scale {exp.reflow.guidance_scale:.2f}, using guidance_scale {guidance_scale:.2f}")
        exp.reflow.guidance_scale = guidance_scale
    is_encodec = isinstance(exp.feature_extractor, rfwave.feature_extractors.EncodecFeatures)

    N = 1 if getattr(exp, 'distill', False) else N

    tot_cost = 0.
    tot_dur = 0.
    if Path(wav_dir).is_dir():
        wav_fps = Path(wav_dir).rglob("*.wav")
    elif Path(wav_dir).is_file() and Path(wav_dir).suffix == '.scp':
        arc_dict = kaldiio.load_scp(wav_dir, max_cache_fd=32)
        wav_fps = arc_dict.items()
    else:
        raise ValueError(f"wav_dir should be a dir or a scp file, got {wav_dir}")

    for wav_fp in tqdm(list(wav_fps)):
        if isinstance(wav_fp, Path):
            y, fs = torchaudio.load(str(wav_fp))
            fn = wav_fp.name
        elif isinstance(wav_fp, tuple):
            fn = wav_fp[0].replace('/', '_') + '.wav'
            fs, y = wav_fp[1]
            y = torch.from_numpy(y.T.astype('float32'))
        else:
            raise ValueError(f"wav_fp should be a Path or a tuple, got {wav_fp}")
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, fs, [["norm", "-3.0"]])
        # y, _ = torchaudio.sox_effects.apply_effects_tensor(y, fs, [["norm", "-1.5"]])

        if y.size(0) > 1:
            y = y[:1]

        rel_dir = wav_fp.relative_to(wav_dir).parent
        save_dir_ = Path(save_dir) / rel_dir
        save_dir_.mkdir(exist_ok=True, parents=True)

        y = y.to(exp.device)
        if is_encodec:
            fn = fn.rstrip('.wav')
            with amp.autocast(enabled=ENABLE_FP16, dtype=torch.float16) and torch.no_grad():
                recon, cost, rvm_loss = copy_synthesis_encodec(exp, y, N=N)
            for k, v in recon.items():
                fn_ = f'{fn}-{k}.wav'
                save_fp = Path(save_dir_) / fn_
                soundfile.write(save_fp, v.astype(float), fs, 'PCM_16')
            for k in cost.keys():
                dur = len(recon[k]) / fs
                tot_dur += dur
                tot_cost += cost[k]
        else:
            save_fp = Path(save_dir_) / fn
            with amp.autocast(enabled=ENABLE_FP16, dtype=torch.float16) and torch.no_grad():
                recon, cost, rvm_loss = copy_synthesis(exp, y, N=N)
            soundfile.write(save_fp, recon.astype(float), fs, 'PCM_16')
            dur = len(recon) / fs
            tot_cost += cost
            tot_dur += dur

    return tot_cost, tot_dur


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--guidance_scale', type=float, default=None)
    parser.add_argument('--num_steps', type=int, default=10)

    args = parser.parse_args()
    assert not (args.model_dir is None and args.pretrained is None)
    Path(args.save_dir).mkdir(exist_ok=True)
    cost, dur = voc(args.model_dir, args.wav_dir, args.save_dir, args.guidance_scale, args.num_steps)
    print(f"Total cost: {cost:.2f}s, Total duration: {dur:.2f}s, ratio: {dur / cost:.2f}")
