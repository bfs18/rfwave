import soundfile
import kaldiio
import numpy as np

from tqdm import tqdm
from pathlib import Path


def default_filename_func(fp):
    # return Path(fp).with_suffix('').name
    return Path(fp).name


def build_ark(filelist, save_prefix, filename_func=default_filename_func,
              segment_dur=2., samples_per_ark=1000):
    if samples_per_ark == 0:
        samples_per_ark = len(filelist)

    shard_scps = []
    for ark_idx, shard_start in enumerate(range(0, len(filelist), samples_per_ark)):
        if shard_start == len(filelist) // samples_per_ark:
            shard_filelist = filelist[shard_start:]
        else:
            shard_filelist = filelist[shard_start: shard_start + samples_per_ark]
        shard_prefix = f'{save_prefix}-{ark_idx:0>3d}'
        save_ark = str(Path(shard_prefix).with_suffix('.ark'))
        save_scp = str(Path(shard_prefix).with_suffix('.scp'))
        shard_scps.append(save_scp)
        writer = kaldiio.WriteHelper(
            f'ark,scp:{save_ark},{save_scp}', write_function='soundfile')
        for wav_fp in tqdm(shard_filelist):
            if not Path(wav_fp).exists():
                continue
            audio = soundfile.read(wav_fp, dtype='int16')  # 16 bit to save space.
            y, sr = audio
            print(audio[0].shape)
            fn = filename_func(wav_fp)
            if y.shape[0] / sr > segment_dur > 0.:
                dur = y.shape[0] / sr
                n = int(dur // segment_dur)
                for i in range(n - 1):
                    seg = y[int(i*segment_dur*sr): int((i+1)*segment_dur*sr)]
                    seg_fn = f'{fn}_{i:0>2d}'
                    writer(seg_fn, (seg, sr))
                seg = y[int((n-1)*segment_dur*sr):]
                fn = f'{fn}_{n-1:0>2d}'
                writer(fn, (seg, sr))
            else:
                writer(fn, audio)
        writer.close()

    scp_lines = []
    for scp in shard_scps:
        scp_lines.extend([l for l in Path(scp).open()])
    Path(save_prefix).with_suffix('.scp').write_text(''.join(scp_lines))


def load_ark_scp(scp_fp):
    ark_dict = kaldiio.load_scp(scp_fp)
    return ark_dict


def test():
    filelist = list(Path('/data/corpus/chs_song/opencpop/wav_mono_48k_16b_norm-3db').glob('*.wav'))[:105]
    save_prefix = '/data/corpus/chs_song/opencpop/test_kaldi_io'
    build_ark(filelist, save_prefix, default_filename_func)
    scp_fp = save_prefix + '.scp'
    ark_dict = load_ark_scp(scp_fp)
    print(list(ark_dict.keys()))
    for k, v in ark_dict.items():
        print(k)
        sr, y = v
        print(sr, y.shape)
        # soundfile.write(k + '.wav', y, sr, 'PCM_16')


if __name__ == "__main__":
    test()
