# Future SST prediction using Diffusion Models
## Dependencies

This project uses both Conda and Poetry to manage dependencies.
To install dependencies for the project, make sure that you have conda installed.

First, create virtual environment managed by conda:

> conda create -f environment.yml

This will create an environment named `sst`.
Then, you need to activate this environment and install other dependencies:

> conda activate sst

> poetry install

## Run Notebooks

To run notebooks on Carbonate's or Bigred200's GPU nodes,
you have to request for a gpu node using `slurm`.
Once the resource is allocated,
you can use the scripts in `scripts/ws` to start jupyter server on the node:

> scripts/ws/carbonate.sh <login_node> <jupyter_server_port>

## Dataset

The project will use Sea Surface Temperature data from [NOAA](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html).

To download the data, just run this script:

> scripts/download_noaa_oi_sst_v2.py

~But the dataset downloaded has resolution of .25deg x .25deg,
which is too detailed for our purpose.
So, we will downscale it to 1deg x 1deg by running the script:~
On a second thought, we don't need to do this!

> scripts/downscale_noaa_ai_sst.py

Next, we will have to extract 29 patches as described in the [paper](https://arxiv.org/abs/1711.07970),
which is:
![](resources/images/sst_prediction_roi.png)

In order to do this, I have guessed the positions of these patches.
Of course, my guesses are not exactly the same
(you can look into the notebook `experiments/guess_patches_location.py`),
but I think it will be good enough.
One more thing, my guesses have 30 patches instead of just 29.

## References
