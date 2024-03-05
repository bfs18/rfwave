import librosa
import soundfile
import torch
import yaml
import time
import torchaudio
import rfwave
import numpy as np

from pathlib import Path
from argparse import ArgumentParser

from inference_voc import load_config, create_instance, load_model


def dur(model_dir, text_lines, phone2id, scale, num_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(model_dir, device=device, last=True)

    phone_info = {}
    for k, line in text_lines.items():
        token_ids = torch.tensor([phone2id[str(tk)] for tk in line.split()])
        token_ids_ = token_ids.unsqueeze(0).repeat_interleave(num_samples, dim=0).to(exp.device)
        features = exp.input_adaptor(token_ids_)
        durations = exp.reflow.sample_ode(features, N=10)[-1].mean(0)[0]
        durations = (durations * scale).round().long()
        durations[durations < 0] = 0
        start_frame = torch.tensor(0, dtype=torch.long, device=exp.device)
        start_phone_idx = torch.tensor(0, dtype=torch.long, device=exp.device)
        phone_info[k] = [token_ids, durations, start_phone_idx, start_frame]
    return phone_info


def get_random_ref(ref_audio, hop_length, padding):
    ctx_n_frame = np.random.randint(200, 300)
    ctx_start_frame = np.random.randint(0, ref_y.size(1) // hop_length - ctx_n_frame)
    ctx_start = ctx_start_frame * hop_length
    ctx_end = (ctx_start_frame + ctx_n_frame) * hop_length
    y_ctx = ref_audio[0, ctx_start: ctx_end]
    ctx_n_frame = torch.tensor(ctx_n_frame + 1 if padding == 'center' else ctx_n_frame)
    return y_ctx, ctx_n_frame


def tts(model_dir, phone_info, save_dir, ref_audio, sr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(model_dir, device=device, last=True)
    config_yaml = Path(model_dir) / 'config.yaml'
    config = load_config(config_yaml)
    hop_length = config['data']['init_args']['train_params']['hop_length']
    padding = config['data']['init_args']['train_params']['padding']
    for k, phone_info in phone_info.items():
        y_ctx, ctx_n_frame = get_random_ref(ref_audio, hop_length, padding)
        y_ctx = y_ctx
        ctx_n_frame = torch.tensor(ctx_n_frame)
        phone_info = [v.unsqueeze(0).to(device) for v in phone_info + [y_ctx, ctx_n_frame]]
        phone_info = exp.process_context(phone_info)
        cond = exp.input_adaptor(*phone_info)
        mel, wave = exp.reflow.sample_ode(cond, N=10)
        mel = mel[-1].detach().cpu()
        wave = wave[-1].detach().cpu()
        torch.save(mel, Path(save_dir) / f'{k}.th')
        soundfile.write(Path(save_dir) / f'{k}-syn.wav', wave[0], sr, 'PCM_16')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dur_model_dir', type=str, required=True)
    parser.add_argument('--tts_model_dir', type=str, required=True)
    parser.add_argument('--ref_audio', type=str, required=True)
    parser.add_argument('--phoneset', type=str, required=True)
    parser.add_argument('--test_txt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()
    Path(args.save_dir).mkdir(exist_ok=True)

    phoneset = torch.load(args.phoneset)
    phoneset = ["_PAD_"] + phoneset
    phone2id = dict([(p, i) for i, p in enumerate(phoneset)])

    ref_y, sr = torchaudio.load(args.ref_audio)

    lines = dict([l.strip().split('|') for l in Path(args.test_txt).open()])
    phone_info = dur(args.dur_model_dir, lines, phone2id, 240/256, num_samples=8)
    tts(args.tts_model_dir, phone_info, args.save_dir, ref_y, sr)
