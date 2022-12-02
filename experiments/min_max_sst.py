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
#   title: Minimum and Maximum SST
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

# +
from __future__ import annotations

import numpy as np

from sst.dataset import NOAA_OI_SST
# -

# # Minimum and Maximum SST
#
# This notebook is to find the maximum and minimum values of SST
# in training patches.
# So that we can use it to scale the temperature into [0, 1] range.

# +
train_ds = NOAA_OI_SST(train=True)

min_temp = np.inf
max_temp = -np.inf
for i in range(len(train_ds)):
    X, _ = train_ds[i]
    p_mintemp = X.min()
    p_maxtemp = X.max()

    if p_mintemp < min_temp:
        min_temp = p_mintemp

    if p_maxtemp > max_temp:
        max_temp = p_maxtemp


print(f'{min_temp=} {max_temp=}')
