import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# # testing training data
# mel1 = torch.randn(1, 100, 40)
# mel2 = torch.cat((torch.randn(1, 90, 40), torch.zeros(1, 10, 40)), dim=1)
# inp = torch.cat((mel1, mel2), dim=0)

# text1 = torch.randint(0, 26, (64,))
# print(text1.shape)
# text2 = torch.randint(0, 26, (54,))
# print(text2.shape)

# text = [text1, text2]

# lens = torch.tensor([100, 90])


# print(inp.shape)
# print(text)
# print(lens)

# batch, seq_len = 2, 100
# text = pad_sequence(text, padding_value=-1, batch_first=True)
# print(text)
# print(text.shape)

def lens_to_mask(t, length=None):
    if length is None:
        length = t.amax()
    seq = torch.arange(length)
    return seq[None, :] < t[:, None]
  
# print(lens.shape)
# print(lens[:, None].shape)      
# mask = lens_to_mask(lens, seq_len)
# print(mask.shape)
# print(mask)

# frac_lengths = torch.zeros((batch, )).float().uniform_(0.7, 1)
# print(frac_lengths)

# def mask_from_frac_lengths(lens, frac_lengths):
#     lengths = lens * frac_lengths
#     max_start = lens - lengths
#     print(max_start)
#     rand = torch.randn(len(frac_lengths))
#     start = max_start * rand
#     end = start + lengths
#     print(start)
#     print(end)
    
#     seq = torch.arange(seq_len)
#     start_mask = seq[None, :] >= start[:, None]
#     end_mask = seq[None, :] < end[:, None]
#     return start_mask & end_mask

# rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
# print(rand_span_mask)
    
# rand_span_mask &= mask
# print(rand_span_mask)

# print(rand_span_mask.shape)
# print(rand_span_mask[..., None].shape)
# print(inp.shape)

# x1 = inp
# x0 = torch.randn_like(x1)

# time = torch.rand((batch,))

# t = time.unsqueeze(-1).unsqueeze(-1)
# φ = (1 - t) * x0 + t * x1
# flow = x1 - x0

# cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

# print("\n\n")
# print(rand_span_mask)
# print("\n")
# print(x1)
# print("\n")
# print(cond[0])
# print("\n")
# print(cond[1])




cond = torch.randn(1, 400, 100)
ref_dys_audio = torch.randn(1, 540, 100)
gen_dys_audio = torch.randn(1, 600, 100)
clean_pad_embed = torch.randn(100)

assert cond.shape[0] == ref_dys_audio.shape[0] == gen_dys_audio.shape[0] == 1, f"batch size 1 is only supported but found {cond.shape[0]}"
batch, seq_len, dtype, device = *cond.shape[:2], cond.dtype, cond.device


############################################################################################
# only works when clean ref audio is shorter than dysarthric ref audio and batch size is 1 #
############################################################################################
# duration
duration = seq_len + (seq_len / ref_dys_audio.shape[1] * gen_dys_audio.shape[1])
duration = torch.tensor(duration, dtype=torch.long)
lens = torch.tensor(seq_len, dtype=torch.long)
# if isinstance(duration, int):
#     duration = torch.full((batch, ), duration, dtype = torch.long)
print(duration)
print(lens)
duration = torch.maximum(duration, lens + 1)
print(duration)
dys_audio = torch.cat((ref_dys_audio, gen_dys_audio), dim=1)
print(dys_audio.shape)
dys_max_len = dys_audio.shape[1]
assert duration <= dys_max_len # shouldn't use assert during batch inference

# create cond mask
cond_mask = torch.ones((1, lens), dtype=torch.bool)
print(cond_mask.shape)
cond_mask = F.pad(cond_mask, (0, duration-lens), value=False)
cond_mask = cond_mask.unsqueeze(-1)
print(cond_mask.shape)
# pad cond with zeros till duration then pad with filler embed till dys_max_len
cond = F.pad(cond, (0, 0, 0, (duration-seq_len)), value=0.0)
print("\n\nworks upto this point ... ")
print(cond.shape)
print(cond_mask.shape)
step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
print(step_cond.shape)
pad_size = dys_max_len - duration
step_cond = torch.cat((cond, clean_pad_embed.repeat(1, pad_size, 1)), dim=1)
print(step_cond.shape)
cond_mask = F.pad(cond_mask, (0, 0, 0, dys_max_len-cond_mask.shape[1], 0, 0), value=False)
print(cond_mask.shape)
print("shapes till here\n\n")

# print(cond)
# print(cond_mask)
