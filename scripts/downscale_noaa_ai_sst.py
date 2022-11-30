#!/usr/bin/env python3

"""
This script will downsample the NOAA OI SST v2 data from .25deg x .25deg to 1deg x 1deg,
the output will be written to `NOAA/OI_SST_v2_1deg/`
"""

import argparse
from collections import namedtuple
import glob
from multiprocessing import Pool
import os
from tqdm import tqdm
import xarray as xr


DownsampleArgs = namedtuple('DownsampleArgs', ['inpath', 'outpath', 'force'])


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        default=False,
        help='Whether to force downsample data files even there is already a downsampled version.')

    return parser.parse_args(args)


def downsample_sst(args: DownsampleArgs):
    outpath = args.outpath

    # Make sure we don't do thing unnecessary.
    if os.path.isfile(outpath) and not args.force:
        return

    ds = xr.load_dataset(args.inpath, engine='netcdf4')

    # Downsample dataset to 1deg x 1deg.
    ds = ds.coarsen(lat=4, lon=4).mean()

    # Save file.
    ds.to_netcdf(outpath, format='NETCDF4')


def main(args=None):
    args = parse_arguments(args)

    # Ensure that the folder exists.
    sst_dir = 'NOAA/OI_SST_v2/'
    assert os.path.isdir(sst_dir), 'Directory `{sst_dir}` doesnt exist.\nPlease execute `scripts/download_noaa_oi_sst_v2.py` first!'

    # List files to be processed.
    files = glob.glob(os.path.join(sst_dir, '*.nc'))

    # Create output directory.
    outdir = 'NOAA/OI_SST_v2_1deg/'
    os.makedirs(outdir, exist_ok=True)

    # Prepare arguments to be processed in parallel.
    downsample_args = (DownsampleArgs(
        inpath=f,
        outpath=os.path.join(outdir, os.path.basename(f)),
        force=args.force) for f in files)

    with Pool(4) as pool:
        tasks = pool.imap_unordered(downsample_sst, downsample_args)

        # Execute each function in parallel.
        for _ in tqdm(tasks, total=len(files)):
            pass


if __name__ == '__main__':
    main()
