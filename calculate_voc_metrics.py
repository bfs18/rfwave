import numpy as np
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_wav_dir', type=str, required=True)
    parser.add_argument('--syn_wav_dir', type=str, required=True)
    parser.add_argument('--sr', type=int, default=24000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rvm = RelativeVolumeMel(sample_rate=args.sr, num_aggregated_bands=3)
    rvm = rvm.to(device)
    utmos_model = UTMOSScore(device=device)

    wav_fps = Path(args.syn_wav_dir).glob('*.wav')
    cnt = 0
    tot_utmos = 0.
    tot_gt_utmos = 0.
    tot_pesq = 0.
    tot_rmv_loss = defaultdict(lambda: 0.)
    for wav_fp in wav_fps:
        gt_fp = Path(args.gt_wav_dir) / wav_fp.name
        syn_y, sr = librosa.load(wav_fp, sr=None)
        gt_y, sr = librosa.load(gt_fp, sr=None)
        min_l = np.minimum(syn_y.shape[0], gt_y.shape[0])
        syn_y = torch.from_numpy(syn_y[..., :min_l]).to(device)
        gt_y = torch.from_numpy(gt_y[..., :min_l]).to(device)
        syn_y_16_khz = torchaudio.functional.resample(syn_y, orig_freq=sr, new_freq=16000)
        gt_y_16_khz = torchaudio.functional.resample(gt_y, orig_freq=sr, new_freq=16000)

        rmv_loss = rvm(syn_y, gt_y)
        syn_utmos_score = utmos_model.score(syn_y_16_khz).mean()
        gt_utmos_score = utmos_model.score(gt_y_16_khz).mean()
        pesq_score = pesq(16000, gt_y.cpu().numpy(), syn_y.cpu().numpy(), 'wb', on_error=1)
        print(rmv_loss, syn_utmos_score, gt_utmos_score, pesq_score)
        for k, v in rmv_loss.items():
            tot_rmv_loss[k] += v.item()
        tot_utmos += syn_utmos_score.item()
        tot_gt_utmos += gt_utmos_score.item()
        tot_pesq += pesq_score
        cnt += 1
    for k, v in tot_rmv_loss.items():
        print(f'{k}: {v/cnt}', end=', ')
    print(f'UTMOS: {tot_utmos/cnt}, PESQ: {tot_pesq/cnt}, GT_UTMOS: {tot_gt_utmos/cnt}')
