import random
import librosa

from argparse import ArgumentParser
from pathlib import Path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wav_dir', required=True)
    parser.add_argument('--alignment_dir', required=True)
    parser.add_argument('--filelist', required=True)
    parser.add_argument('--valid_ratio', type=float, default=1/64)

    args = parser.parse_args()
    random.seed(123456)

    alignment_fps = Path(args.alignment_dir).glob("*.th")
    filelist = []
    for alignment_fp in alignment_fps:
        fn = alignment_fp.stem
        wav = Path(args.wav_dir) / f"{fn}.wav"
        dur = librosa.get_duration(filename=wav)
        filelist.append('|'.join([fn, str(wav), str(alignment_fp), f"{dur:.2f}"]))
    random.shuffle(filelist)
    num_valid = int(len(filelist) * args.valid_ratio)
    num_train = len(filelist) - num_valid
    train_filelist = sorted(filelist[:num_train])
    valid_filelist = sorted(filelist[num_train:])
    Path(args.filelist + '.train').write_text('\n'.join(train_filelist) + '\n')
    Path(args.filelist + '.valid').write_text('\n'.join(valid_filelist) + '\n')

