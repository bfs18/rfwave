import random

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from pydub import AudioSegment
from bounded_pool_executor import BoundedProcessPoolExecutor
from ark_io import build_ark


random.seed(123456)


def split_artists(raw_list):
    artists = []
    for l in raw_list:
        artists.append(l.split()[1])
    artists = sorted(list(set(artists)))
    num = len(artists)
    num_valid = int(num * 0.02)
    num_test = int(num * 0.02)
    num_train = num - num_valid - num_test
    random.shuffle(artists)
    artists_valid = artists[:num_valid]
    artists_test = artists[num_valid: num_valid+num_test]
    artists_train = artists[num_valid+num_test:]
    return artists_train, artists_valid, artists_test


def make_filelist(artists, raw_list, wav_dir):
    filelist = []
    for l in raw_list:
        segs = l.split()
        artist = segs[1]
        path = segs[3]
        if artist in artists:
            wav_fp = f'{wav_dir}/{path.replace("mp3", "wav")}'
            filelist.append(wav_fp)
    return filelist


def mp3_to_wav(mp3_fp, wav_fp):
    audio = AudioSegment.from_mp3(mp3_fp)
    audio.export(wav_fp, format='wav')


def convert_corpus(mp3_dir, wav_dir):
    mp3_fps = list(Path(mp3_dir).glob('*/*.mp3'))
    executor = BoundedProcessPoolExecutor(max_workers=32)
    futures = []
    for mp3_fp in tqdm(mp3_fps):
        artist_dir_name = mp3_fp.parent.name
        save_dir = Path(wav_dir) / artist_dir_name
        save_dir.mkdir(exist_ok=True, parents=True)
        wav_fp = save_dir / mp3_fp.with_suffix('.wav').name
        futures.append(executor.submit(mp3_to_wav, mp3_fp, wav_fp))
    [f.result() for f in futures]


def filename_func(fp):
    wav_fn = Path(fp).name
    artist_name = Path(fp).parent.name
    return f'{artist_name}/{wav_fn}'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--raw_tsv', required=True)
    parser.add_argument('--mp3_dir', required=True)
    parser.add_argument('--wav_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--segment_dur', type=float, default=10.)
    args = parser.parse_args()

    # convert_corpus(args.mp3_dir, args.wav_dir)
    raw_list = [l.strip() for l in Path(args.raw_tsv).open()][1:]
    train, valid, test = split_artists(raw_list)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    for artists, name in zip(
            [train, valid, test], ['train', 'valid', 'test']):
        filelist = make_filelist(artists, raw_list, args.wav_dir)
        filelist_fp = save_dir / f'{name}.txt'
        filelist_fp.write_text('\n'.join(filelist) + '\n')
        ark_save_prefix = f'{args.save_dir}/jamendo-{name}'
        print(f'building ark {ark_save_prefix}')
        build_ark(filelist, ark_save_prefix, filename_func, segment_dur=args.segment_dur)
