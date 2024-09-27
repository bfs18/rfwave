import numpy as np
import os
import torch
import librosa
import torchaudio
import numpy as np

from argparse import ArgumentParser
from pesq import pesq
from metrics.UTMOS import UTMOSScore
from rfwave.rvm import RelativeVolumeMel
from pathlib import Path
from collections import defaultdict
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
from metrics.periodicity import calculate_periodicity_metrics
from tqdm import tqdm


def create_visqol_api(mode):
    assert mode in ['audio', 'speech']
    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

    api = visqol_lib_py.VisqolApi()

    api.Create(config)
    return api


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_wav_dir', type=str, required=True)
    parser.add_argument('--syn_wav_dir', type=str, required=True)
    parser.add_argument('--mode', required=True, choices=['audio', 'speech'])
    parser.add_argument('--sr', type=int, default=24000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rvm = RelativeVolumeMel(sample_rate=args.sr, num_aggregated_bands=3)
    rvm = rvm.to(device)
    utmos_model = UTMOSScore(device=device)
    visqol_api = create_visqol_api(args.mode)

    wav_fps = Path(args.syn_wav_dir).rglob('*.wav')
    cnt = 0
    tot_utmos = 0.
    tot_gt_utmos = 0.
    tot_pesq = 0.
    tot_visqol = 0.
    tot_periodicity = 0.
    tot_f1 = 0.
    tot_rmv_loss = defaultdict(lambda: 0.)
    for wav_fp in tqdm(list(wav_fps)):
        gt_fp = Path(args.gt_wav_dir) / wav_fp.relative_to(args.syn_wav_dir)
        syn_y, sr = librosa.load(wav_fp, sr=None)
        gt_y, sr = librosa.load(gt_fp, sr=None)
        min_l = np.minimum(syn_y.shape[0], gt_y.shape[0])
        syn_y = torch.from_numpy(syn_y[:min_l]).to(device)
        gt_y = torch.from_numpy(gt_y[:min_l]).to(device)
        syn_y_16_khz = torchaudio.functional.resample(syn_y, orig_freq=sr, new_freq=16000)
        gt_y_16_khz = torchaudio.functional.resample(gt_y, orig_freq=sr, new_freq=16000)
        periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(
            gt_y_16_khz.unsqueeze(0), syn_y_16_khz.unsqueeze(0))
        tot_periodicity += periodicity_loss
        tot_f1 += f1_score
        rmv_loss = rvm(syn_y, gt_y)
        syn_utmos_score = utmos_model.score(syn_y_16_khz).mean()
        gt_utmos_score = utmos_model.score(gt_y_16_khz).mean()
        pesq_score = pesq(16000, gt_y_16_khz.cpu().numpy(), syn_y_16_khz.cpu().numpy(), 'wb', on_error=1)
        # print(rmv_loss, syn_utmos_score, gt_utmos_score, pesq_score)
        for k, v in rmv_loss.items():
            tot_rmv_loss[k] += v.item()

        if args.mode == 'audio':
            visqol_ref = gt_y_16_khz.detach().cpu().numpy()
            visqol_deg = syn_y_16_khz.detach().cpu().numpy()
        else:
            syn_y_48_khz = torchaudio.functional.resample(syn_y, orig_freq=sr, new_freq=48000)
            gt_y_48_khz = torchaudio.functional.resample(gt_y, orig_freq=sr, new_freq=48000)
            visqol_ref = gt_y_48_khz.detach().cpu().numpy()
            visqol_deg = syn_y_48_khz.detach().cpu().numpy()
        visqol = visqol_api.Measure(visqol_ref.astype(np.float64), visqol_deg.astype(np.float64))

        tot_utmos += syn_utmos_score.item()
        tot_gt_utmos += gt_utmos_score.item()
        tot_pesq += pesq_score
        tot_visqol += visqol.moslqo
        cnt += 1
    for k, v in tot_rmv_loss.items():
        print(f'{k}: {v/cnt:.2f}', end=', ')
    print(f'UTMOS: {tot_utmos/cnt:.2f}, PESQ: {tot_pesq/cnt:.2f}, '
          f'GT_UTMOS: {tot_gt_utmos/cnt:.2f}, VISQOL: {tot_visqol/cnt:.2f} '
          f'V/UV F1 {tot_f1/cnt:.2f}, Periodicity: {tot_periodicity/cnt:.2f}')
