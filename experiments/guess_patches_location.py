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
#   title: Guess Patches Location
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

# +
from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# -

# # Guess Patches Location

ds = xr.load_dataset(
    'NOAA/OI_SST_v2/sst.day.mean.2006.nc', engine='netcdf4')
lat = ds['lat'].values
lon = ds['lon'].values
sst = ds['sst'].values

# +
def plot_sst(sst: np.ndarray, lat: np.ndarray, lon: np.ndarray, domain: tuple[float, float, float, float] | None = None, figsize=None):
    if figsize is None:
        figsize = (12, 6)

    xx, yy = np.meshgrid(lon, lat)

    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        subplot_kw=dict(projection=crs),
        figsize=figsize)
    ax.coastlines()
    ax.contourf(xx, yy, sst)

    if domain is not None:
        ax.set_extent(domain, crs=crs)

    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    fig.tight_layout()
    return fig, ax

_ = plot_sst(sst[0], lat, lon)
# -

# ## Find Region of Interest

# Ok, now we will zoom in the Atlantic ocean to have something similar in the
# [paper](https://arxiv.org/abs/1711.07970),
# which is: ![](../resources/images/sst_prediction_roi.png)

_ = plot_sst(sst[0], lat, lon, domain=(-75, -10, 25, 65), figsize=(8, 6))

# OK, it looks like that the domain of interest is from $75^\circ W$ to $10^\circ W$,
# and from $25^\circ N$ to $65^\circ N$.

# ## Find Patches' Location

# Now, we will work with the region we found.

# +
ds_roi = ds.sel(lat=slice(25, 65), lon=slice(285, 350))
lat_roi = ds_roi['lat'].values
lon_roi = ds_roi['lon'].values
sst_roi = ds_roi['sst'].values

_ = plot_sst(sst_roi[0], lat_roi, lon_roi, figsize=(8, 6))
# -

def plot_patches(ds_roi: xr.Dataset, patches_pos: list[tuple[float, float]], patch_size: float):
    lat = ds_roi['lat'].values
    lon = ds_roi['lon'].values
    sst = ds_roi['sst'].values[0]

    _, ax = plot_sst(sst, lat, lon, figsize=(8, 6))
    coord_system = ccrs.PlateCarree()

    for plat, plon in patches_pos:
        y = [plat, plat + patch_size, plat + patch_size, plat, plat]
        x = [plon, plon, plon + patch_size, plon + patch_size, plon]
        ax.plot(x, y, color='black', lw=1, transform=coord_system)

# ### First Row Patches

# By using eyeballing and guesstimate techniques,
# we have the following positions for the first 9 patches:

patch_size = 6 # in degrees
first_row_patches_position = [
    # (lat, lon) also in degrees
    (30, -64),
    (30, -59),
    (30, -54),
    (30, -49),
    (30, -44),
    (30, -39),
    (30, -34),
    (30, -29),
    (30, -24),
]
plot_patches(ds_roi, first_row_patches_position, patch_size)

# It's not the best guesstimate, but I think it will work good enough.

# ### Second Row Patches
#
# Moving on with the next 7 patches: from patch #10 to patch #16.

second_row_patches_position = [
    (35, -64),
    (35, -59),
    (35, -54),
    (35, -49),
    (35, -44),
    (35, -39),
    (35, -24),
]
plot_patches(ds_roi, second_row_patches_position, patch_size)

# ### Third Row Patches
#
# Moving on with the next 7 patches: from patch #17 to patch #23.

third_row_patches_position = [
    (40, -59),
    (40, -54),
    (40, -49),
    (40, -44),
    (40, -39),
    (40, -34),
    (40, -29),
    (40, -24),
]
plot_patches(ds_roi, third_row_patches_position, patch_size)

# Here, due to my guess is not very accurate,
# we can have up to 8 patches.

# ### Last Row Patches

last_row_patches_position = [
    # (45, -64),
    # (45, -59),
    # (45, -54),
    (45, -49),
    (45, -44),
    (45, -39),
    (45, -34),
    (45, -29),
    (45, -24),
]
plot_patches(ds_roi, last_row_patches_position, patch_size)

# ### All Patches
#
# So in total, we can extract 30 patches.
# The locations of these patches are:

all_patches = (first_row_patches_position
    + second_row_patches_position
    + third_row_patches_position
    + last_row_patches_position)
all_patches

plot_patches(ds_roi, all_patches, patch_size)
