from pathlib import Path

import librosa


def select_short(filelist, thres=15.):
    new_filelist = []
    for l in filelist:
        d = librosa.get_duration(path=l.split('|')[1])
        if d < thres:
            new_filelist.append(l)
    return new_filelist


filelist_fp = Path('')
filelist = [l.strip() for l in Path(filelist_fp).open()]
new_filelist = select_short(filelist)
Path(filelist_fp.with_name('short-' + filelist_fp.name)).write_text('\n'.join(new_filelist) + '\n')
