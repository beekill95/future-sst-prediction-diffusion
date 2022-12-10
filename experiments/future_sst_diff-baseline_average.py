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
#   title: Future SST Difference Prediction using Average of Previous Diff As Baseline
#     Diffusion
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

# +
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sst.dataset import NOAA_OI_SST, Difference_NOAA_OI_SST
# -

# # Future SST Difference Prediction using Average of Previous Diff As Baseline
#
# In this experiment,
# I will construct a baseline to compare with diffusion model
# by using the average of previous difference as the prediction for the future difference.
#
# In this,
# we just need the test dataset as we don't need to train anything.

# +
test_ds = NOAA_OI_SST(train=False)
test_diff_ds = Difference_NOAA_OI_SST(test_ds)

# Use dataloader just to make these evaluations
# as close as possible to what we used in diffusion models.
test_dataloader = DataLoader(test_ds, batch_size=256, num_workers=4)
test_diff_dataloader = DataLoader(test_diff_ds, batch_size=256, num_workers=4)

total_diff_err = 0.
total_temp_err = 0.
for (X_diff, y_diff), (X, y) in zip(test_diff_dataloader, test_dataloader):
    # Predict the future difference by using the previous differences.
    y_pred_diff = torch.mean(X_diff, dim=1, keepdim=True)

    # Compare the predicted difference with the true prediction.
    total_diff_err += float(F.mse_loss(y_pred_diff, y_diff).item())

    # Get back the original temperature.
    y_pred = X[:, -1:] + y_pred_diff

    # Compare the predicted temperature with the true temperature.
    total_temp_err += float(F.mse_loss(y_pred, y))

# Display the result.
print(f'Difference error: {(total_diff_err / len(test_dataloader)):.4f}')
print(f'Temperature error: {(total_temp_err / len(test_dataloader)):.4f}')
