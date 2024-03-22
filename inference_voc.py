import librosa
import warnings
import soundfile
import torch
import yaml
import time
import rfwave
import re
import kaldiio
import torchaudio

from pathlib import Path
from argparse import ArgumentParser


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
    if "reflow.backbone.module.pos_embed" in model_dict['state_dict']:
        del model_dict['state_dict']["reflow.backbone.module.pos_embed"]
    exp.load_state_dict(model_dict['state_dict'])
    exp.eval()
    exp.to(device)
    return exp


def copy_synthesis(exp, y, N=1000):
    features = exp.feature_extractor(y)
    start = time.time()
    sample = exp.reflow.sample_ode(features, N=N)[-1]
    cost = time.time() - start
    l = min(sample.size(-1), y.size(-1))
    rvm_loss = exp.rvm(sample[..., :l], y[..., :l])
    recon = sample.detach().cpu().numpy()[0]
    return recon, cost, rvm_loss


def voc(model_dir, wav_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(model_dir, device=device, last=True)

    tot_cost = 0.
    tot_dur = 0.
    if Path(wav_dir).is_dir():
        wav_fps = Path(wav_dir).rglob("*.wav")
    elif Path(wav_dir).is_file() and Path(wav_dir).suffix == '.scp':
        arc_dict = kaldiio.load_scp(wav_dir, max_cache_fd=32)
        wav_fps = arc_dict.items()
    else:
        raise ValueError(f"wav_dir should be a dir or a scp file, got {wav_dir}")

    for wav_fp in wav_fps:
        if isinstance(wav_fp, Path):
            y, fs = torchaudio.load(wav_fp)
            fn = wav_fp.name
        elif isinstance(wav_fp, tuple):
            fn = wav_fp[0].replace('/', '_') + '.wav'
            fs, y = wav_fp[1]
            y = torch.from_numpy(y.T.astype('float32'))
        else:
            raise ValueError(f"wav_fp should be a Path or a tuple, got {wav_fp}")

        if y.size(0) > 1:
            y = y[:1]

        y = y.to(exp.device)
        save_fp = Path(save_dir) / fn
        recon, cost, rvm_loss = copy_synthesis(exp, y, N=10)
        soundfile.write(save_fp, recon, fs, 'PCM_16')
        dur = len(recon) / fs
        tot_cost += cost
        tot_dur += dur

    return tot_cost, tot_dur


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()
    Path(args.save_dir).mkdir(exist_ok=True)
    cost, dur = voc(args.model_dir, args.wav_dir, args.save_dir)
    print(f"Total cost: {cost:.2f}s, Total duration: {dur:.2f}s, ratio: {cost / dur:.2f}")
