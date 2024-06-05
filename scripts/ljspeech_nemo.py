import torch
import string

from collections import Counter
from pathlib import Path
from nemo_text_processing.text_normalization.normalize import Normalizer
from g2p_en import G2p


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

meta_csv = 'LJSpeech-1.1/metadata.csv'
save_dir = 'LJSpeech-1.1/synta_alignment'

meta_lines = [l.strip() for l in Path(meta_csv).open()]
meta_dict = dict()
for l in meta_lines:
    fields = l.split('|')
    # field[2] is normalized text
    meta_dict[fields[0]] = fields[2]

phone_dict = dict()
final_phone_dict = dict()
phoneset = Counter()
for k, v in meta_dict.items():
    phone_dict[k] = g2p(v)
    temp_phone = remove_space_around_punctuation(phone_dict[k])
    final_phone_dict[k] = add_bos_eos(temp_phone)
    print(k, "#phones", len(final_phone_dict[k]))
    phoneset.update(final_phone_dict[k])
    save_fp = Path(save_dir) / f'{k}.th'
    torch.save({'tokens': final_phone_dict[k]}, save_fp)

print([phoneset.most_common()])
phoneset = sorted(list(phoneset.keys()))
print(len(phoneset), phoneset)

torch.save(phoneset, Path(save_dir).parent / 'phoneset.th')
