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
#   title: Future SST Difference Prediction using Fully Connected Networks with Conditional
#     Diffusion
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

# +
from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
from tqdm.auto import tqdm

from sst import (
    ConditionalBackwardSampler,
    ForwardSampler,
    LinearBetaScheduler,
)
from sst.dataset import NOAA_OI_SST, Difference_NOAA_OI_SST
from sst.dataset.transforms import ScaleTemperature
from sst.layers import SinusoidalPositionEncoding
from sst.training_procedures import ConditionalOnPastSSTTrainingProcedure
# -

# # Future SST Difference Prediction using Fully Connected Networks with Conditional Diffusion
#
# In this experiment,
# I will try to implement the idea of Saharia et al. (2022) to predict future SST
# conditioning on the past 4-day SST.
# But instead of using UNet architecture, I will just use a normal fully-connected network.

# ## Dataset
#
# Similar to that of [sst_unconditional_diffusion.py](./sst_unconditional_diffusion.py):

temp_diff_scaler = ScaleTemperature(min_temp=0, max_temp=1)
transform_fn = Compose([
    # temp_diff_scaler,
    # Lambda(lambda s: (2*s[0] - 1, 2*s[1] - 1))
])
train_ds = NOAA_OI_SST(train=True)
train_ds = Difference_NOAA_OI_SST(
    train_ds, transform=transform_fn)

# We will want to split the data into training and validation data using 20% split:
# 80% for training and 20% for validation.

train_size = int(len(train_ds) * 0.8)
val_size = len(train_ds) - train_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size])
print(f'{train_size=} {len(train_ds)=}')
print(f'{val_size=} {len(val_ds)=}')

# In addition, we will need test dataset for this experiment.

test_ds = NOAA_OI_SST(train=False)
test_diff_ds = Difference_NOAA_OI_SST(
    test_ds, transform=transform_fn)
print(f'{len(test_ds)=}')


# ## Fully-Connected Network

class FutureSSTNoisePredictionModel(nn.Module):
    def __init__(self, nb_past_observations: int) -> None:
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

        self.past_observation_mlp = nn.Sequential(
            nn.Linear(24*24*nb_past_observations, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
        )

        self.hidden = nn.Sequential(
            # Encoder.
            nn.Linear(4096, 2048), nn.LayerNorm(2048), nn.SiLU(),
            # # Decoder.
            nn.Linear(2048, 24 * 24),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 24, 24))


    def forward(self, noisy_sst: torch.Tensor, past_sst: torch.Tensor, time_steps: torch.Tensor):
        # Process noisy SST.
        x1 = self.flatten(noisy_sst)
        x1 = self.first(x1)
        x1 += self.time_mlp(time_steps)

        # Process past SST.
        x2 = self.flatten(past_sst)
        x2 = self.past_observation_mlp(x2)

        # Concatenate the noisy SST and past SST.
        x = torch.concat([x1, x2], dim=1)

        x = self.hidden(x)
        x = self.unflatten(x)
        return x


# ### Training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# +
beta_scheduler = LinearBetaScheduler(1e-4, 0.02, 1024)
forward_sampler = ForwardSampler(beta_scheduler)

# I didn't verify whether the forward sampler is doing what we expect.
# So let's do it now.
time_steps = [1, 4, 8, 32, 64, 128, 256, 512, 1023]
fig, axes = plt.subplots(
    ncols=len(time_steps) + 1,
    figsize=(4 * len(time_steps) + 4, 4))
for ax, t in zip(axes[1:], time_steps):
    _, sst = train_ds[2555]
    sst = torch.tensor(sst)
    noisy_sst, _ = forward_sampler(sst, torch.tensor(t))
    cs = ax.imshow(noisy_sst[0])
    fig.colorbar(cs, ax=ax)
    ax.axis('off')
    ax.set_title(f'{t=}')

ax = axes[0]
cs = ax.imshow(sst[0])
fig.colorbar(cs, ax=ax)
ax.axis('off')
ax.set_title('Original')
fig.tight_layout()
# -

# It looks good!
#
# Now, we will train the model.

# + tags=[]
model = FutureSSTNoisePredictionModel(3)
training_procedure = ConditionalOnPastSSTTrainingProcedure(model, forward_sampler, device)

train_dataloader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_ds, batch_size=256, num_workers=4)

epochs = 1 if device == 'cpu' else 500
for epoch in range(epochs):
    train_loss = training_procedure.train(train_dataloader, epoch)
    val_loss = training_procedure.evaluate(val_dataloader)

    print(f'{train_loss=} {val_loss=}')
# -

# ### Results
#
# Now, let's check how does the model work.

# +
backward_sampler = ConditionalBackwardSampler(model, beta_scheduler, device)

past_sst, original_sst = train_ds[500]
past_sst, original_sst = torch.tensor(past_sst), torch.tensor(original_sst)
noisy_sst, _ = forward_sampler(
    original_sst, torch.tensor(forward_sampler.max_time_steps - 1))
recovered_ssts = []
recovered_sst = noisy_sst
for i in reversed(range(forward_sampler.max_time_steps)):
    recovered_sst = backward_sampler(recovered_sst, past_sst, i)

    if i % 200 == 0:
        recovered_ssts.append(recovered_sst)

nb_sst = len(recovered_ssts) + 1
fig, axes = plt.subplots(ncols=nb_sst, figsize=(4 * nb_sst, 4))
for t, (ax, sst) in enumerate(zip(axes, recovered_ssts)):
    ax.set_title(f'{t=}')
    cs = ax.pcolormesh(sst[0])
    fig.colorbar(cs, ax=ax)
    ax.axis('off')

ax = axes[-1]
cs = ax.pcolormesh(original_sst[0])
fig.colorbar(cs, ax=ax)
ax.set_title('Original SST Field')
ax.axis('off')
fig.tight_layout()

print(F.mse_loss(sst, original_sst))
# -

# #### Evaluate MSE with Test Dataset

# +
shown = False

@torch.no_grad()
def evaluate_mse_for_diff_ds(
        *,
        diff_dl: DataLoader,
        original_dl: DataLoader,
        backward_sampler: ConditionalBackwardSampler,
        max_time_steps: int,
        diff_temp_scaler: ScaleTemperature):
    assert len(diff_dl) == len(original_dl)

    nb_batches = len(diff_dl)
    diff_total_error = 0.
    temp_total_error = 0.

    for (X_diff, y_diff), (X, y) in tqdm(zip(diff_dl, original_dl), total=nb_batches):
        # Perform backward process.
        y_pred_diff = torch.randn_like(y_diff)

        # Feed the noisy diff sst into backward sampler to get original diff sst.
        for t in reversed(range(max_time_steps)):
            y_pred_diff = backward_sampler(y_pred_diff, X_diff, t)

        # Convert to the normal range.
        y_diff = diff_temp_scaler.inverse(y_diff)
        y_pred_diff = diff_temp_scaler.inverse(y_pred_diff.detach().cpu())

        diff_total_error += float(F.mse_loss(y_diff, y_pred_diff).item())

        # Calculate the predicted temperature.
        temp_pred = X[:, -1:] + y_pred_diff
        temp_total_error += float(F.mse_loss(y, temp_pred).item())

        # Plot results.
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        ax = axes[0, 0]
        cs = ax.imshow(y_pred_diff[0, 0])
        ax.set_title('Predicted temp difference')
        fig.colorbar(cs, ax=ax)

        ax = axes[0, 1]
        cs = ax.imshow(y_diff[0, 0])
        ax.set_title('True temp difference')
        fig.colorbar(cs, ax=ax)

        ax = axes[1, 0]
        cs = ax.imshow(temp_pred[0, 0])
        ax.set_title('Predicted temp')
        fig.colorbar(cs, ax=ax)

        ax = axes[1, 1]
        cs = ax.imshow(y[0, 0])
        ax.set_title('Original temp')
        fig.colorbar(cs, ax=ax)
        fig.suptitle('Diff')
        fig.tight_layout()

    return dict(
        diff_err=diff_total_error / nb_batches,
        temp_err=temp_total_error / nb_batches,
    )


test_dataloader = DataLoader(test_ds, batch_size=256, num_workers=4)
test_diff_dataloader = DataLoader(test_diff_ds, batch_size=256, num_workers=4)
evaluate_mse_for_diff_ds(
    diff_dl=test_diff_dataloader,
    original_dl=test_dataloader,
    backward_sampler=backward_sampler,
    max_time_steps=forward_sampler.max_time_steps,
    diff_temp_scaler=temp_diff_scaler)
