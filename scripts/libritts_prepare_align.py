import os
import shutil

import torchaudio

from pathlib import Path

SAVE_FLAC = False
SAVE_LAB = False
CREATE_LK = False
CREATE_FILELIST = True

libritts_dir = Path('/data2/liupeng/corpus/LibriTTS')
save_dir = Path('/data2/liupeng/corpus/LibriTTS-align')
flac_dir = Path('/data2/liupeng/corpus/LibriTTS-output')
datasets = ['dev-clean', 'dev-other' 'test-clean', 'test-other',
            'train-clean-100', 'train-clean-360', 'train-other-500']

for dataset in datasets:
    spkr_dirs = (libritts_dir / dataset).glob('*')
    dataset_fids = []
    for spkr_dir in spkr_dirs:
        print(str(spkr_dir))
        save_spkr_dir = save_dir / spkr_dir.name
        flac_spkr_dir = flac_dir / spkr_dir.name
        if CREATE_LK:
            save_spkr_dir.mkdir(exist_ok=True, parents=True)
        if SAVE_FLAC:
            flac_spkr_dir.mkdir(exist_ok=True, parents=True)
        wav_fps = spkr_dir.rglob('*.wav')
        for wav_fp in wav_fps:
            txt_fp = wav_fp.with_suffix('.normalized.txt')
            if CREATE_LK:
                lk_wav_fp = save_spkr_dir / wav_fp.name
                lk_txt_fp = save_spkr_dir / f'{wav_fp.stem}.lab'
                os.symlink(wav_fp, lk_wav_fp)
                os.symlink(txt_fp, lk_txt_fp)
            if SAVE_FLAC:
                flac_wav_fp = flac_spkr_dir / f'{wav_fp.stem}.flac'
                y, sr = torchaudio.load(str(wav_fp))
                torchaudio.save(str(flac_wav_fp), y, sr)
            if SAVE_LAB:
                lab_fp = flac_spkr_dir / f'{wav_fp.stem}.lab'
                shutil.copy(txt_fp, lab_fp)
            dataset_fids.append(wav_fp.stem)
    if CREATE_FILELIST:
        filelist_fp = flac_dir / f'{dataset}_fids.txt'
        filelist_fp.write_text('\n'.join(sorted(dataset_fids)) + '\n')
