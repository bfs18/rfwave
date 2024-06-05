import os
import shutil
import string
import torchaudio
import torch

from collections import Counter
from pathlib import Path
from nemo_text_processing.text_normalization.normalize import Normalizer
from g2p_en import G2p

SAVE_FLAC = False
SAVE_LAB = False
CREATE_LK = False
CREATE_FILELIST = False
CREATE_PHONES = True


def remove_space_around_punctuation(phones):
    new_phones = []
    for i in range(len(phones)):
        if phones[i] in string.punctuation:
            if len(new_phones) and new_phones[-1] == ' ':
                del new_phones[-1]
        elif i > 0 and phones[i - 1] in string.punctuation and phones[i] == ' ':
            continue
        new_phones.append(phones[i])
    return new_phones


def add_bos_eos(phones):
    new_phones = []
    if phones[-1] not in string.punctuation:
        new_phones = phones + ['.']
    else:
        new_phones = phones[:]
    return ["<BOS>"] + new_phones


# text_normalizer = Normalizer(input_case="cased", lang="en")
g2p = G2p()


libritts_dir = Path('/data2/liupeng/corpus/LibriTTS')
save_dir = Path('/data2/liupeng/corpus/LibriTTS-align')
flac_dir = Path('/data2/liupeng/corpus/LibriTTS-output')
datasets = ['dev-clean', 'dev-other' 'test-clean', 'test-other',
            'train-clean-100', 'train-clean-360', 'train-other-500']

phoneset = Counter()

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
            if CREATE_PHONES:
                text = txt_fp.read_text().strip()
                phones = g2p(text)
                phones = remove_space_around_punctuation(phones)
                phones = add_bos_eos(phones)
                print(wav_fp.stem, "#phones", len(phones))
                phoneset.update(phones)
                save_fp = flac_spkr_dir / f'{wav_fp.stem}.space.th'
                torch.save({'tokens': phones}, save_fp)
            dataset_fids.append(wav_fp.stem)
    if CREATE_FILELIST:
        filelist_fp = flac_dir / f'{dataset}_fids.txt'
        filelist_fp.write_text('\n'.join(sorted(dataset_fids)) + '\n')

print("phone set:")
print([phoneset.most_common()])
phoneset = sorted(list(phoneset.keys()))
print(len(phoneset), phoneset)

torch.save(phoneset, flac_dir / 'space_phoneset.th')
