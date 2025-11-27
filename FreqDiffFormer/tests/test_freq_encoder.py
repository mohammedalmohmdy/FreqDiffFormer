
import torch
from models.freq_encoder import DCTFrequencyEncoder
m = DCTFrequencyEncoder(in_ch=1, emb_dim=256, use_topk=0.5)
m.eval()
x = torch.rand(2,1,64,64)  # batch of 2 sketches
with torch.no_grad():
    out = m(x)
print("Output shape:", out.shape)
