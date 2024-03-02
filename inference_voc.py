import librosa
import soundfile
import torch
import yaml
import time
import rfwave

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


def load_model(model_dir, device):
    config_yaml = Path(model_dir) / 'config.yaml'
    ckpt_fp = list(Path(model_dir).rglob("last.ckpt"))
    if len(ckpt_fp) == 0:
        raise ValueError(f"No checkpoint found in {model_dir}")
    ckpt_fp = ckpt_fp[0]

    config = load_config(config_yaml)
    exp = create_instance(config['model'])

    model_dict = torch.load(ckpt_fp, map_location='cpu')
    exp.load_state_dict(model_dict['state_dict'])
    exp.to(device)
    return exp


def copy_synthesis(exp, wav_fp, save_fp, N=1000):
    y, fs = librosa.load(wav_fp, sr=None)
    y = torch.from_numpy(y)
    y = y.to(exp.device)
    y = y.unsqueeze(0)
    features = exp.feature_extractor(y)
    start = time.time()
    sample = exp.reflow.sample_ode(features, N=N)[-1]
    cost = time.time() - start
    l = min(sample.size(-1), y.size(-1))
    rvm_loss = exp.rvm(sample[..., :l], y[..., :l])
    recon = sample.detach().cpu().numpy()[0]
    soundfile.write(save_fp, recon, fs, 'PCM_16')
    return cost, recon.shape[0] / fs, rvm_loss


def voc(model_dir, wav_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(model_dir, device=device)

    tot_cost = 0.
    tot_dur = 0.
    for wav_fp in Path(wav_dir).rglob("*.wav"):
        save_fp = Path(save_dir) / wav_fp.name
        cost, dur, rvm_loss = copy_synthesis(exp, wav_fp, save_fp, N=10)
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
