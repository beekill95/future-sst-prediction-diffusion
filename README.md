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
