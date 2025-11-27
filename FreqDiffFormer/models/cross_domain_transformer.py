import torch
import torch.nn as nn

class CrossDomainTransformer(nn.Module):
    def __init__(self, dim=256, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, sketch_embed, img_embed):
        # simple concatenation-based cross-attention via transformer encoder
        x = torch.stack([sketch_embed, img_embed], dim=0)  # (2, B, D)
        out = self.transformer(x)  # (2, B, D)
        return out[0], out[1]
