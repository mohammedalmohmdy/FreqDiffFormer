
import torch
import torch.nn as nn
import math

def timestep_embedding(timesteps, dim):
    # sinusoidal embedding
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = torch.cat([emb, torch.zeros(len(timesteps), 1)], dim=1)
    return emb

class LatentUNet(nn.Module):
    def __init__(self, latent_dim=512, hidden=1024, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.time_emb_dim = time_emb_dim

    def forward(self, x, t):
        """
        x: (B, latent_dim)
        t: (B,) long tensor
        """
        temb = timestep_embedding(t, self.time_emb_dim).to(x.device)
        temb = self.time_mlp(temb)
        h = self.net[0](x)
        h = self.net[1](h)
        h = h + temb  # simple conditioning
        h = self.net[2:](h)
        return h
