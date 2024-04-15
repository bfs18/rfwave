import torch
import librosa

from pathlib import Path


flac_dir = Path('/data2/liupeng/corpus/LibriTTS-output')
datasets = ['dev-clean', 'dev-other' 'test-clean', 'test-other',
            'train-clean-100', 'train-clean-360', 'train-other-500']


phoneset = set()
phoneset_fp = flac_dir / "phoneset.th"
for dataset in datasets:
    fid_fp = flac_dir / f'{dataset}-fids.txt'
    filelist_fp = flac_dir / f'{dataset}.filelist'
    filelist = []
    fids = set([l.strip() for l in fid_fp.open()])
    for fid in fids:
        spk_dir = fid.split('_')[0]
        audio_fp = flac_dir / spk_dir / f'{fid}.flac'
        align_fp = flac_dir / spk_dir / f'{fid}.th'
        if audio_fp.exists() and align_fp.exists():
            dur = librosa.get_duration(path=audio_fp)
            tokens = torch.load(align_fp)['tokens']
            filelist.append('|'.join([fid, str(audio_fp), str(align_fp), f'{dur:.2f}']))
            phoneset.update(tokens)
    filelist_fp.write_text('\n'.join(filelist) + '\n')
phoneset = sorted(list(phoneset))
torch.save(phoneset, phoneset_fp)
