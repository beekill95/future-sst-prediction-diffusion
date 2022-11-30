#!/usr/bin/env python3

"""
This script will download NOAA OI SST v2 into `NOAA` folder.
"""

import argparse
import asyncio
import httpx
import os
import tqdm.asyncio


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--year-range',
        dest='year_range',
        type=int,
        default=[2006, 2017],
        nargs=2,
        help='Years range to download. Default from 2006 to 2017.'
    )
    parser.add_argument(
        '--force-download',
        dest='force_download',
        action='store_true',
        default=False,
        help='Force download even if SST for a year is downloaded.',
    )

    return parser.parse_args(args)


async def download_daily_sst(years: list[int], outputdir: str, force_download: bool):
    async def save_to_file(client: httpx.AsyncClient, url: str, dest: str):
        if os.path.isfile(dest) and not force_download:
            return

        async with client.stream('get', url) as response:
            with open(dest, 'wb') as outfile:
                async for data in response.aiter_bytes():
                    outfile.write(data)

    # A sample SST file is located at:
    # https://downloads.psl.noaa.gov//Datasets/noaa.oisst.v2.highres/sst.day.mean.1981.nc
    HOST = 'https://downloads.psl.noaa.gov/'
    FRAGMENT = 'Datasets/noaa.oisst.v2.highres'
    urls = [(f'{HOST}/{FRAGMENT}/sst.day.mean.{y}.nc',
             os.path.join(outputdir, f'sst.day.mean.{y}.nc')) for y in years]

    async with httpx.AsyncClient() as client:
        tasks = (save_to_file(client, url, dest) for url, dest in urls)
        await tqdm.asyncio.tqdm_asyncio.gather(*tasks)


def main(args=None):
    args = parse_arguments(args)

    # Obtain years to download.
    year_from, year_end = args.year_range
    years = list(range(year_from, year_end + 1))

    # Create output directory.
    outdir = os.path.join('NOAA/OI_SST_v2')
    os.makedirs(outdir, exist_ok=True)

    # Download SST.
    asyncio.run(download_daily_sst(years, outdir, args.force_download))


if __name__ == '__main__':
    main()
