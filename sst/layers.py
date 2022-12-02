import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        encodings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        encodings = torch.exp(torch.arange(half_dim, device=device) * -encodings)
        encodings = time[:, None] * encodings[None, :]
        encodings = torch.cat((encodings.sin(), encodings.cos()), dim=-1)
        return encodings
