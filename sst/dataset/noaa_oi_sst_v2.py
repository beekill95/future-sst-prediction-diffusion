from __future__ import annotations


import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Iterator
import xarray as xr


PATCHES_DIR = 'NOAA/OI_SST_v2_patches/'
TRAIN_PATCHES = list(range(1, 17)) + list(range(21, 31))
TRAIN_YEARS = list(range(2006, 2016))
TEST_PATCHES = [17, 18, 19, 20]
TEST_YEARS = [2016, 2017]

TransformFn = Callable[[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]


class NOAA_OI_SST(Dataset):
    def __init__(self, train: bool, previous_days: int = 4, transform: TransformFn | None = None) -> None:
        super().__init__()

        # Load the desired patches.
        indices, years = (TRAIN_PATCHES, TRAIN_YEARS) if train else (TEST_PATCHES, TEST_YEARS)
        patch_datasets = load_patches(indices, years)

        XX, yy = [], []
        for patch_ds in patch_datasets:
            X, y = extract_Xy(patch_ds, previous_days)
            XX.append(X)
            yy.append(y)

            # TODO: how can we normalize and standardize these?
            # Like they did in the paper?

        self._X = np.concatenate(XX, axis=0)
        self._y = np.concatenate(yy, axis=0)
        self._transform = transform

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, index):
        X, y = self._X[index], self._y[index]

        if self._transform:
            return self._transform((X, y))

        return X, y


class Residual_NOAA_OI_SST(Dataset):
    def __init__(self, noaa_ds: Dataset) -> None:
        super().__init__()
        self._ds = noaa_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        X, y = self._ds[idx]

        residual = y - X[-1]
        return X, residual


class Difference_NOAA_OI_SST(Dataset):
    def __init__(self, noaa_ds: Dataset, transform: TransformFn) -> None:
        super().__init__()
        self._ds = noaa_ds
        self._transform = transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        X, y = self._ds[idx]

        diff = X[1:] - X[:-1]
        target = y - X[-1]

        transform_fn = self._transform
        if transform_fn is not None:
            diff, target = transform_fn((diff, target))

        return diff, target


def load_patches(patch_indices, years) -> Iterator[xr.Dataset]:
    def is_valid_patch_index(filepath: str) -> bool:
        filename, _ = os.path.splitext(os.path.basename(filepath))
        patch_part = filename.split('.')[-1]
        return int(patch_part[-2:]) in patch_indices

    def is_valid_patch_year(filepath: str) -> bool:
        filename, _ = os.path.splitext(os.path.basename(filepath))
        year_part = filename.split('.')[-2]
        return int(year_part) in years

    assert os.path.isdir(PATCHES_DIR), f'Dataset directory {PATCHES_DIR} doesnt exist. Please run `scripts/extract_sst_patches.py` first!'

    files = glob.iglob(os.path.join(PATCHES_DIR, '*.nc'))
    files = filter(is_valid_patch_year, files)
    files = list(filter(is_valid_patch_index, files))
    # File names themselves are sortable.
    files = sorted(files)

    for f in files:
        yield xr.load_dataset(f, engine='netcdf4')


def extract_Xy(patch_ds: xr.Dataset, previous_days: int) -> tuple[np.ndarray, np.ndarray]:
    # sst will have shape of (time, lat, lon)
    sst = patch_ds['sst'].values

    XX = []
    yy = []
    for i in range(previous_days, sst.shape[0]):
        # Using SST of previous days;
        X = sst[i-previous_days:i]
        # to predict current SST.
        y = sst[i][None, ...]

        XX.append(X[None, ...])
        yy.append(y[None, ...])

    return np.concatenate(XX, axis=0), np.concatenate(yy, axis=0)
