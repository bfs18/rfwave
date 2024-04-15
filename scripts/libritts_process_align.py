import textgrid
import string
import re
import torch

from pathlib import Path
from collections import OrderedDict


def split_punct(lab):
    lab = ''.join([c for c in lab if c not in '"()'])
    words = []
    for w in lab.split():
        if '-' in w:
            words.extend(w.split('-'))
        else:
            words.append(w)
    words_with_punc = []
    for w in words:
        nw = [w]
        left_punct_m = re.match(rf"^[{string.punctuation}]+", nw[0])
        if left_punct_m is not None:
            left_punct = left_punct_m.group()
            nw[0] = nw[0][len(left_punct):]
            nw.insert(0, left_punct[:1])
        right_punct_m = re.search(rf"[{string.punctuation}]+$", nw[-1])
        if right_punct_m is not None:
            right_punc = right_punct_m.group()
            nw[-1] = nw[-1][:-len(right_punc)]
            nw.append(right_punc[:1])
        words_with_punc.extend(nw)
    return [w for w in words_with_punc if w != '']


def is_punct(word):
    return all([p in string.punctuation for p in word])


def process_alignment(tg, lab):
    word_intervals = tg.tiers[0]
    phone_intervals = tg.tiers[1]

    if '[bracketed]' in [word_int.mark for word_int in word_intervals]:
        return None

    words_with_punc = split_punct(lab)
    print(words_with_punc)
    print([word_int.mark for word_int in word_intervals])

    none_time_dict = OrderedDict()
    lab_idx, int_idx = 0, 0
    while int_idx < len(word_intervals):
        wl = words_with_punc[lab_idx] if lab_idx < len(words_with_punc) else "."
        int_word = word_intervals[int_idx]
        if int_word.mark in ['', ')']:
            if is_punct(wl):
                lab_idx += 1
                none_time_dict[(int_word.minTime, int_word.maxTime)] = wl
            else:
                none_time_dict[(int_word.minTime, int_word.maxTime)] = (
                    "SIL" if int_idx == 0 or int_idx == len(word_intervals) - 1 else "SP")
            int_idx += 1
        else:
            wi = int_word.mark
            if is_punct(wl):
                lab_idx += 1
            wl = words_with_punc[lab_idx] if lab_idx < len(words_with_punc) else "."
            # print(wi, wl)
            assert re.sub(r'["\(\)-]+', '', wi.lower()) == wl.lower() or (is_punct(wl) and is_punct(wi))
            int_idx += 1
            lab_idx += 1

    # print([pi.mark for pi in phone_intervals])
    for phone_int in phone_intervals:
        time_tup = (phone_int.minTime, phone_int.maxTime)
        if time_tup in none_time_dict:
            phone_int.mark = none_time_dict[time_tup]
    # print([pi.mark for pi in phone_intervals])
    return phone_intervals


def process_alignment_default(tg):
    word_intervals = tg.tiers[0]
    phone_intervals = tg.tiers[1]

    if '[bracketed]' in [word_int.mark for word_int in word_intervals]:
        return None

    for int_idx, phone_int in enumerate(phone_intervals):
        if phone_int.mark in ["", "spn"]:
            phone_int.mark = (
                "SIL" if int_idx == 0 or int_idx == len(phone_intervals) - 1 else "SP")
    return phone_intervals


libritts_dir = "/Users/liupeng/wd_disk/dataset/LibriTTS-output"
audio_fps = Path(libritts_dir).rglob("*.flac")

for audio_fp in audio_fps:
    lab_fp = audio_fp.with_suffix(".lab")
    tg_fp = audio_fp.with_suffix(".TextGrid")
    save_fp = audio_fp.with_suffix(".th")
    if lab_fp.exists() and tg_fp.exists():
        print(lab_fp)
        tg = textgrid.TextGrid.fromFile(tg_fp)
        lab = lab_fp.read_text().strip()
        try:
            phone_intervals = process_alignment(tg, lab)
        except:
            phone_intervals = process_alignment_default(tg)
        if phone_intervals is None:
            continue
        tokens = [pi.mark for pi in phone_intervals]
        durations = [pi.maxTime - pi.minTime for pi in phone_intervals]
        torch.save({"tokens": tokens, "durations": durations}, save_fp)


