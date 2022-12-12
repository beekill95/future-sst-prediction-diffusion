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
#   title: Baseline Future SST Difference Prediction using Fully Connected Networks
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
from tqdm.autonotebook import tqdm

from sst.dataset import NOAA_OI_SST, Difference_NOAA_OI_SST
# -

# # Baseline Future SST Difference Prediction using Fully Connected Networks
#
# In this notebook,
# we will establish a baseline for future SST difference prediction using fully connected networks.
# So no fancy diffusion models here.
# We just want to know whether a very simple network can do the job or not.

# ## Dataset
#
# Similar to that of [sst_unconditional_diffusion.py](./sst_unconditional_diffusion.py):

train_ds = NOAA_OI_SST(train=True)
train_ds = Difference_NOAA_OI_SST(train_ds)

# We will want to split the data into training and validation data using 20% split:
# 80% for training and 20% for validation.

train_size = int(len(train_ds) * 0.8)
val_size = len(train_ds) - train_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size])
print(f'{train_size=} {len(train_ds)=}')
print(f'{val_size=} {len(val_ds)=}')

# In addition, we will need test dataset for this experiment.

test_ds = NOAA_OI_SST(train=False)
test_diff_ds = Difference_NOAA_OI_SST(test_ds)
print(f'{len(test_ds)=}')


# ## Fully-Connected Network

class BaselineSSTDifferenceModel(nn.Module):
    def __init__(self, nb_past_observations: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(24*24*nb_past_observations, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 2048), nn.LayerNorm(2048), nn.ReLU(), nn.Dropout(0.5),
        )
        self.output = nn.Linear(2048, 24*24)
        self.unflatten = nn.Unflatten(1, unflattened_size=(1, 24, 24))

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.output(x)
        x = self.unflatten(x)
        return x

# ### Training

# + tags=[]
class BaselineTrainingProcedure:
    def __init__(self, model: BaselineSSTDifferenceModel, device: str, lr: float = 1e-3) -> None:
        self._model = model.to(device)
        self._device = device

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

    def train(self, dataloader: DataLoader, epoch: int) -> float:
        model = self._model.train()
        device = self._device
        optimizer = self._optimizer

        total_loss = 0.
        nb_batches = len(dataloader)
        for X, y in tqdm(dataloader, total=nb_batches, desc=f'Training epoch #{epoch}'):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = F.smooth_l1_loss(y_pred, y)

            total_loss += float(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss /  nb_batches

    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        model = self._model.eval()
        device = self._device

        total_loss = 0.
        nb_batches = len(dataloader)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            # loss = F.mse_loss(y_pred, y)
            loss = F.smooth_l1_loss(y_pred, y)

            total_loss += float(loss.cpu().item())

        return total_loss / nb_batches


# Obtain training device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Training model.
model = BaselineSSTDifferenceModel(3)
train_dataloader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_ds, batch_size=256, num_workers=4)

training_procedure = BaselineTrainingProcedure(model, device)
epochs = 1 if device == 'cpu' else 100
for epoch in range(epochs):
    train_loss = training_procedure.train(train_dataloader, epoch)
    val_loss = training_procedure.evaluate(val_dataloader)

    print(f'{train_loss=:.4f} {val_loss=:.4f}')
# -

# ### Results
#
# Here, we will display some prediction results.

test_idx = [100, 250, 233, 750]
for idx in test_idx:
    X, y = test_diff_ds[idx]
    X = torch.tensor(X[None, ...]).to(device)
    y_pred = model(X).detach().cpu()

    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    ax = axes[0]
    cs = ax.pcolormesh(y[0])
    fig.colorbar(cs, ax=ax)
    ax.set_title('Original SST Difference')

    ax = axes[1]
    cs = ax.pcolormesh(y_pred[0, 0])
    fig.colorbar(cs, ax=ax)
    ax.set_title('Predicted SST Difference')

    fig.suptitle(f'test_diff_ds[{idx}]')
    fig.tight_layout()


# ## Test MSE

# +
model.eval()
test_dataloader = DataLoader(test_ds, batch_size=256, num_workers=4)
test_diff_dataloader = DataLoader(test_diff_ds, batch_size=256, num_workers=4)

diff_total_err = 0.
temp_total_err = 0.

with torch.no_grad():
    for (X_diff, y_diff), (X, y) in zip(test_diff_dataloader, test_dataloader):
        y_diff_pred = model(X_diff.to(device)).detach().cpu()

        # Calculate mse.
        diff_total_err += float(F.mse_loss(y_diff_pred, y_diff).item())

        # Convert to normal temperature.
        y_pred = X[:, -1:] + y_diff_pred
        temp_total_err += float(F.mse_loss(y_pred, y).item())

print(f'Difference error: {(diff_total_err / len(test_dataloader)):.4f}')
print(f'Temperature error: {(temp_total_err / len(test_dataloader)):.4f}')

# +
# Train difference error.
model.eval()
diff_total_err = 0.
temp_total_err = 0.

with torch.no_grad():
    for X_diff, y_diff in train_dataloader:
        y_diff_pred = model(X_diff.to(device)).detach().cpu()

        # Calculate mse.
        diff_total_err += float(F.mse_loss(y_diff_pred, y_diff).item())

print(f'Difference error: {(diff_total_err / len(train_dataloader)):.4f}')

# +
# Validation difference error.
model.eval()
diff_total_err = 0.
temp_total_err = 0.

with torch.no_grad():
    for X_diff, y_diff in val_dataloader:
        y_diff_pred = model(X_diff.to(device)).detach().cpu()

        # Calculate mse.
        diff_total_err += float(F.mse_loss(y_diff_pred, y_diff).item())

print(f'Difference error: {(diff_total_err / len(val_dataloader)):.4f}')
