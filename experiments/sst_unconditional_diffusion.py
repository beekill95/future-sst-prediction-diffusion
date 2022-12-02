# ---
# jupyter:
#   author: Quan Nguyen
#   jupytext:
#     formats: py:light,ipynb
#     notebook_metadata_filter: title,author,date
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   title: Unconditional SST Diffusion
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

# +
from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda

from sst import (
    BackwardSampler,
    ForwardSampler,
    LinearBetaScheduler,
)
from sst.dataset import NOAA_OI_SST
from sst.dataset.transforms import ScaleTemperature
from sst.layers import SinusoidalPositionEncoding
from sst.training_procedures import UnconditionalTrainingProcedure
# -

# # Unconditional SST Diffusion
#
# This is just a test to see what kind of error we can expect from a simple diffusion model on SST dataset,
# and to see what a generated SST looks like.

# ## Dataset
#
# First, load data into memory and display some temperature fields.

train_ds = NOAA_OI_SST(
    train=True,
    transform=Compose([
        ScaleTemperature(min_temp=-2, max_temp=36),
        Lambda(lambda s: (2*s[0] - 1, 2*s[1] - 1)),
    ],
))
print(f'{len(train_ds)=}')

# +
def show_sst_fields(ds: NOAA_OI_SST, idx: int):
    X, y = ds[idx]

    nb_fields = X.shape[0] + 1
    fig, axes = plt.subplots(ncols=nb_fields, figsize=(4 * nb_fields, 4))
    for i, (sst, ax) in enumerate(zip(X, axes)):
        ax.set_title(f'X[{i}]')
        cs = ax.pcolormesh(sst)
        fig.colorbar(cs, ax=ax)

    ax = axes[-1]
    ax.set_title('y')
    cs = ax.pcolormesh(y[0])
    fig.colorbar(cs, ax=ax)

    fig.suptitle(f'ds[{idx}]')
    fig.tight_layout()

show_sst_fields(train_ds, 100)
show_sst_fields(train_ds, 200)
show_sst_fields(train_ds, 777)
# -

# ## Noise-Predicting Model

class DiffusionNoiseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEncoding(24 * 24),
            nn.Linear(24*24, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
        )
        self.first = nn.Sequential(
            nn.Linear(24*24, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
        )

        self.hidden = nn.Sequential(
            # Encoder.
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.SiLU(),
            # # Decoder.
            nn.Linear(1024, 2048), nn.LayerNorm(2048), nn.SiLU(),
            nn.Linear(2048, 24 * 24),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 24, 24))

    def forward(self, x, time_steps):
        x = self.flatten(x)
        x = self.first(x)
        x += self.time_mlp(time_steps)
        x = self.hidden(x)
        x = self.unflatten(x)
        return x

# ### Training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# +
beta_scheduler = LinearBetaScheduler(1e-4, 0.2, 1024)
forward_sampler = ForwardSampler(beta_scheduler)

model = DiffusionNoiseModel()
training_procedure = UnconditionalTrainingProcedure(model, forward_sampler, device)

train_dataloader = DataLoader(train_ds, batch_size=256, shuffle=True)
epochs = 1 if device == 'cpu' else 100
for epoch in range(epochs):
    train_loss = training_procedure.train(train_dataloader, epoch)
    print(f'{train_loss=}')
# -

# ### Results

# +
backward_process = BackwardSampler(model, beta_scheduler, device)

_, original_sst = train_ds[1000]
original_sst = torch.tensor(original_sst)
noisy_sst, _ = forward_sampler(
    original_sst, torch.tensor(forward_sampler.max_time_steps - 1))
recovered_ssts = []
recovered_sst = noisy_sst
for i in reversed(range(forward_sampler.max_time_steps)):
    recovered_sst = backward_process.backward(recovered_sst, i)

    if i % 200 == 0:
        recovered_ssts.append(recovered_sst)

nb_sst = len(recovered_ssts) + 1
fig, axes = plt.subplots(ncols=nb_sst, figsize=(4 * nb_sst, 4))
for ax, sst in zip(axes, recovered_ssts):
    cs = ax.pcolormesh(sst[0])
    fig.colorbar(cs, ax=ax)
    ax.axis('off')

ax = axes[-1]
cs = ax.pcolormesh(original_sst[0])
fig.colorbar(cs, ax=ax)
ax.set_title('Original SST Field')
ax.axis('off')
fig.tight_layout()
