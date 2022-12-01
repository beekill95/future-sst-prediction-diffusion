#!/usr/bin/env python3

"""
This script will extract ~29 patches~ 30 patches as described
in the paper: https://arxiv.org/abs/1711.07970
and in `experiments/guess_patches_location.py`
"""

import argparse
from collections import namedtuple
import glob
from multiprocessing import Pool
import os
import shutil
from tqdm import tqdm
import xarray as xr


# These positions are guesstimates based on the figure 4
# (further information is in `experiments/guess_patches_location.py`).
# So, ¯\_(ツ)_/¯
# Also, the positions are lat and lon in degrees.
patches_lower_left_position = [
    # (lat, lon) in degrees
    (30, -64),
    (30, -59),
    (30, -54),
    (30, -49),
    (30, -44),
    (30, -39),
    (30, -34),
    (30, -29),
    (30, -24),
    # Second row.
    (35, -64),
    (35, -59),
    (35, -54),
    (35, -49),
    (35, -44),
    (35, -39),
    (35, -24),
    # Third row.
    (40, -59),
    (40, -54),
    (40, -49),
    (40, -44),
    (40, -39),
    (40, -34),
    (40, -29),
    (40, -24),
    # Last row.
    (45, -49),
    (45, -44),
    (45, -39),
    (45, -34),
    (45, -29),
    (45, -24),
]
domain_size = 6 # also in degrees.


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        default=False,
        help='Whether to force extract patches.')

    parser.add_argument(
        '--processes', '-p',
        default=4,
        help='Number of processes to run the tasks in parallel.')

    return parser.parse_args(args)


ExtractPatchesArgs = namedtuple('ExtractPatchesArgs', ['ds_path', 'outdir', 'patches_pos', 'patch_size'])
def extract_patches(args: ExtractPatchesArgs):
    ds = xr.load_dataset(args.ds_path, engine='netcdf4')
    filename, ext = os.path.splitext(os.path.basename(args.ds_path))

    for patch_idx, (plat, plon) in enumerate(args.patches_pos):
        # Convert lon to 0 - 360.
        plon += 360
        patch_ds = ds.sel(lat=slice(plat, plat + args.patch_size), lon=slice(plon, plon + args.patch_size))

        # Save patch.
        outpath = os.path.join(args.outdir, f'{filename}.patch_{patch_idx + 1:0>2}{ext}')
        patch_ds.to_netcdf(outpath, format='NETCDF4')


def main(args=None):
    args = parse_arguments(args)

    # Make sure input directory exists.
    indir = 'NOAA/OI_SST_v2'
    assert os.path.isdir(indir), f'{indir} doesnt exist. Please execute `scripts/download_noaa_oi_sst_v2.py` first!'

    # Check if outpudir exists or not.
    outdir = 'NOAA/OI_SST_v2_patches'
    if os.path.isdir(outdir):
        if args.force:
            print(f'WARN: {outdir} exists and will be removed due to `force` flag is True.')
            shutil.rmtree(outdir)
        else:
            raise FileExistsError('Output directory {outdir} exists!')

    # Create output directory.
    os.makedirs(outdir)

    # List files.
    files = glob.glob(os.path.join(indir, '*.nc'))

    with Pool(args.processes) as pool:
        tasks = pool.imap_unordered(
            extract_patches,
            (ExtractPatchesArgs(
                ds_path=f,
                outdir=outdir,
                patches_pos=patches_lower_left_position,
                patch_size=domain_size) for f in files))

        # Extract patches in parallel.
        for _ in tqdm(tasks, total=len(files)):
            pass

if __name__ == '__main__':
    main()
