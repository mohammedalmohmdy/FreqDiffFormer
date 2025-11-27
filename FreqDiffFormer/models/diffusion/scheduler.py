
import torch
class LinearBetaSchedule:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    def q_sample(self, x0, t, noise):
        """
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        t: (B,) long tensor with values in [0, timesteps-1]
        """
        a_bar = self.alpha_bar[t].view(-1, *([1]*(x0.dim()-1)))
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
