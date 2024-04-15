import torch

from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict


parser = ArgumentParser()
parser.add_argument('ckpt_path', type=str)
ckpt_path = Path(parser.parse_args().ckpt_path)
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if 'backbone' in k:
        new_state_dict[k.replace('backbone.', 'backbone._orig_mod.')] = v
    elif 'input_adaptor' in k:
        new_state_dict[k.replace('input_adaptor.', 'input_adaptor._orig_mod.')] = v
    else:
        new_state_dict[k] = v
ckpt['state_dict'] = new_state_dict
torch.save(ckpt, ckpt_path.with_name('new_' + ckpt_path.name))
