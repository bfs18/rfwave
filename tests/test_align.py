import torch

from rfwave.dit import calculate_alignment_loss, find_segment_tokens, find_segment_tokens_


attn = torch.rand([1, 12, 10])
start = torch.tensor([12])
token_length = torch.tensor([10])
token_exp_scale = torch.tensor([4])
attn[:, :, 3: 7] *= 10
attn = attn / attn.sum(dim=-1, keepdim=True)
find_segment_tokens_(attn[0], thres=0.5)
s, e = find_segment_tokens(attn[0], thres=0.5)
print(s, e)
loss = calculate_alignment_loss(attn, start, token_length, token_exp_scale)
print(loss)
