import librosa
import soundfile
import torch
import yaml
import time
import torchaudio
import rfwave
import numpy as np
import warnings

from pathlib import Path
from argparse import ArgumentParser
from rfwave.dataset import upsample_durations

from inference_voc import load_config, create_instance, load_model


def dur(model_dir, text_lines, phone2id, scale, num_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(model_dir, device=device, last=True)

    phone_info = {}
    for k, line in text_lines.items():
        token_ids = torch.tensor([phone2id[str(tk)] for tk in line.split()])
        token_ids_ = token_ids.unsqueeze(0).to(exp.device)
        features = exp.input_adaptor(token_ids_)
        durations = exp.reflow.sample_ode(features, N=10)[-1].mean(0)[0]
        durations = (durations * scale).round().long()
        durations[durations < 0] = 0
        start_frame = torch.tensor(0, dtype=torch.long, device=exp.device)
        start_phone_idx = torch.tensor(0, dtype=torch.long, device=exp.device)
        phone_info[k] = [token_ids, durations, start_phone_idx, start_frame]
    return phone_info


def get_random_ref(ref_audio, ref_align, hop_length, padding):
    ctx_n_frame = np.random.randint(200, 300)
    ctx_start_frame = np.random.randint(0, ref_y.size(1) // hop_length - ctx_n_frame)
    ctx_start = ctx_start_frame * hop_length
    ctx_end = (ctx_start_frame + ctx_n_frame) * hop_length
    y_ctx = ref_audio[0, ctx_start: ctx_end]
    ctx_n_frame = torch.tensor(ctx_n_frame + 1 if padding == 'center' else ctx_n_frame)
    if ref_align is not None:
        up_durations = upsample_durations(
            ref_align['durations'], ref_audio.size(1), hop_length, padding)
        cs_durations = torch.cumsum(up_durations, 0)
        ctx_end_frame = ctx_start_frame + ctx_n_frame
        start_phone_idx = torch.searchsorted(cs_durations, ctx_start_frame, right=True)
        end_phone_idx = torch.searchsorted(cs_durations, ctx_end_frame, right=False)
        token_ids = ref_align['tokens'][start_phone_idx: end_phone_idx + 1]
        durations = up_durations[start_phone_idx: end_phone_idx + 1].detach().clone()
        if end_phone_idx != start_phone_idx:
            first_num_frames = cs_durations[start_phone_idx] - ctx_start_frame
            last_num_frames = ctx_end_frame - cs_durations[end_phone_idx - 1]
            durations[0] = first_num_frames
            durations[-1] = last_num_frames
        else:
            durations[0] = ctx_end_frame - ctx_start_frame
    else:
        token_ids, durations = None, None
    return y_ctx, ctx_n_frame, token_ids, durations


def tts(model_dir, phone_info, save_dir, ref_audio, ref_align, sr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(model_dir, device=device, last=True)
    config_yaml = Path(model_dir) / 'config.yaml'
    config = load_config(config_yaml)
    hop_length = config['data']['init_args']['train_params']['hop_length']
    padding = config['data']['init_args']['train_params']['padding']
    for k, phone_info in phone_info.items():
        (y_ctx, ctx_n_frame, ctx_token_ids, ctx_durations
         ) = get_random_ref(ref_audio, ref_align, hop_length, padding)
        y_ctx = y_ctx
        ctx_n_frame = torch.tensor(ctx_n_frame)
        if ctx_token_ids is not None and ctx_durations is not None:
            ref_info = [y_ctx, ctx_n_frame, ctx_token_ids, ctx_durations]
        else:
            ref_info = [y_ctx, ctx_n_frame]
        phone_info = [v.unsqueeze(0).to(device) for v in phone_info + ref_info]
        phone_info = exp.process_context(phone_info)
        cond = exp.input_adaptor(*phone_info)
        start = torch.tensor([0])
        length = torch.tensor([phone_info[1].sum()])
        mel, wave = exp.reflow.sample_ode(cond, N=10, start=start, out_length=length)
        mel = mel[-1].detach().cpu()
        wave = wave[-1].detach().cpu()
        torch.save(mel, Path(save_dir) / f'{k}.th')
        soundfile.write(Path(save_dir) / f'{k}-syn.wav', wave[0], sr, 'PCM_16')


def tts_e2e(model_dir, text_lines, save_dir, ref_audio, sr, num_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(model_dir, device=device, last=True)
    config_yaml = Path(model_dir) / 'config.yaml'
    config = load_config(config_yaml)
    hop_length = config['data']['init_args']['train_params']['hop_length']
    padding = config['data']['init_args']['train_params']['padding']
    for k, line in text_lines.items():
        token_ids = torch.tensor([phone2id[str(tk)] for tk in line.split()])
        y_ctx, ctx_n_frame, *_ = get_random_ref(ref_audio, None, hop_length, padding)
        num_tokens = torch.tensor(token_ids.shape[0])
        ctx_length = torch.tensor(ctx_n_frame)
        phone_info = [token_ids, y_ctx, num_tokens, ctx_length]
        phone_info = [v.unsqueeze(0).to(device) for v in phone_info]
        pi_kwargs = {'num_tokens': phone_info[2], 'ctx_length': phone_info[3]}
        phone_info = phone_info[:2]
        phone_info[1] = exp.feature_extractor(phone_info[1])
        text = exp.input_adaptor(*phone_info)
        dur_info = exp.attn_or_dur(None, text, **pi_kwargs)
        pi_kwargs.update(**dur_info)
        mel, wave = exp.reflow.sample_ode(text, N=100, **pi_kwargs)
        mel = mel[-1].detach().cpu().numpy()[0]
        wave = wave[-1].detach().cpu().numpy()[0]
        torch.save(mel, Path(save_dir) / f'{k}.th')
        soundfile.write(Path(save_dir) / f'{k}-syn.wav', wave, samplerate=sr, subtype='PCM_16')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dur_model_dir', type=str, required=False, default=None)
    parser.add_argument('--tts_model_dir', type=str, required=True)
    parser.add_argument('--ref_audio', type=str, required=True)
    parser.add_argument('--ref_align', type=str, required=False, default=None)
    parser.add_argument('--phoneset', type=str, required=True)
    parser.add_argument('--test_txt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()
    Path(args.save_dir).mkdir(exist_ok=True)

    phoneset = torch.load(args.phoneset)
    phoneset = ["_PAD_"] + phoneset
    phone2id = dict([(p, i) for i, p in enumerate(phoneset)])

    ref_y, sr = torchaudio.load(args.ref_audio)
    if args.ref_align is not None and Path(args.ref_align).exists():
        ref_align = torch.load(args.ref_align)
        ref_align['tokens'] = torch.tensor([phone2id[str(tk)] for tk in ref_align['tokens']])
    else:
        warnings.warn("No reference alignment provided")
        ref_align = None

    lines = dict([l.strip().split('|') for l in Path(args.test_txt).open()])
    if args.dur_model_dir is not None and Path(args.ref_align).exists():
        phone_info = dur(args.dur_model_dir, lines, phone2id, 240/256, num_samples=8)
        tts(args.tts_model_dir, phone_info, args.save_dir, ref_y, ref_align, sr)
    else:
        tts_e2e(args.tts_model_dir, lines, args.save_dir, ref_y, sr)
