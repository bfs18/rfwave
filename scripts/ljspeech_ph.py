import json
import torch
import random
import os

from argparse import ArgumentParser
from pathlib import Path


def load_annotated_json(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    phones = data[0][1]
    durations = torch.tensor([int(d * 100) for d in data[0][4]], dtype=torch.long)
    return {"durations": durations, "tokens": phones}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wav_dir', required=True)
    parser.add_argument('--json_dir', required=True)
    parser.add_argument('--alignment_dir', required=True)
    parser.add_argument('--filelist', required=True)
    parser.add_argument('--valid_ratio', type=float, default=1/64)

    args = parser.parse_args()
    random.seed(123456)
    os.makedirs(args.alignment_dir, exist_ok=True)
    json_fps = Path(args.json_dir).glob("*.json")
    phoneset_fp = Path(args.filelist).parent / "kaldi_phoneset.th"

    filelist = []
    phoneset = set()
    for json_fp in json_fps:
        fn = json_fp.stem
        alignment_fp = Path(args.alignment_dir) / f"{fn}.th"
        wav = Path(args.wav_dir) / f"{fn}.wav"
        data = load_annotated_json(json_fp)
        torch.save(data, alignment_fp)
        filelist.append('|'.join([fn, str(wav), str(alignment_fp)]))
        phoneset.update(data['tokens'])
    phoneset = sorted(list(phoneset))
    random.shuffle(filelist)
    num_valid = int(len(filelist) * args.valid_ratio)
    num_train = len(filelist) - num_valid
    train_filelist = sorted(filelist[:num_train])
    valid_filelist = sorted(filelist[num_train:])
    Path(args.filelist + '.train').write_text('\n'.join(train_filelist) + '\n')
    Path(args.filelist + '.valid').write_text('\n'.join(valid_filelist) + '\n')
    torch.save(phoneset, phoneset_fp)
