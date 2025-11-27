
import torch
from models.diffusion.ddpm import LatentDDPM
device = 'cpu'
ddpm = LatentDDPM(latent_dim=128, timesteps=10, device=device)
ddpm.to(device)
x0 = torch.randn(4, 128, device=device)
loss = ddpm.forward_loss(x0)
print("DDPM forward loss:", loss.item())
# sampling
x_gen = ddpm.sample((4,128), device=device)
print("Sample shape:", x_gen.shape)
