import json
import torch
import random
import os
import numpy as np
import pickle
import librosa
from pathlib import Path
from copy import deepcopy


class IndexedDataset:
    def __init__(self, path, num_cache=1):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f"{path}.idx", allow_pickle=True).item()['offsets']
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1


if __name__ == '__main__':
    valid_ratio = 0.01
    filelist_fp = '/data1/corpus/LJSpeech-1.1/synta_filelist'
    alignment_dir = '/data1/corpus/LJSpeech-1.1/synta_alignment'
    bin_fp = '/data1/corpus/ljspeech'
    phoneset_fp = Path(filelist_fp).parent / "synta_phoneset.th"
    wav_dir = '/data1/corpus/LJSpeech-1.1/wavs'
    os.makedirs(alignment_dir, exist_ok=True)

    random.seed(123456)
    os.makedirs(alignment_dir, exist_ok=True)
    split = ['train', 'valid', 'test']

    phoneset = set()
    for s in split:
        dataset = IndexedDataset(os.path.join(bin_fp, s))
        filelist = []
        for d in dataset:
            fn = d['item_name']
            phones = d['ph'].split()
            duration = torch.tensor(d['dur'], dtype=torch.long)
            alignment_fp = Path(alignment_dir) / f"{fn}.th"
            wav = Path(wav_dir) / f"{fn}.wav"
            dur = librosa.get_duration(path=wav)
            torch.save({'tokens': phones, 'durations': duration}, alignment_fp)
            filelist.append('|'.join([fn, str(wav), str(alignment_fp), str(len(phones)), f'{dur:.2f}']))
            phoneset.update(phones)
        random.shuffle(filelist)
        Path(filelist_fp + f'.{s}').write_text('\n'.join(filelist) + '\n')
    phoneset = sorted(list(phoneset))
    torch.save(phoneset, phoneset_fp)
