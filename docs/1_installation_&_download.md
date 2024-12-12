# Installation and download

- setup the nuPlan dataset following the [offiical-doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)
- setup conda environment
```
conda create -n gpd python=3.9
conda activate gpd

# install nuplan-devkit and navsim
git clone https://github.com/motional/nuplan-devkit.git
git clone https://github.com/autonomousvision/navsim.git
export PYTHONPATH="path/to/nuplan-devkit:path/to/navsim:$PYTHONPATH"

# setup GPD-1
cd ..
git clone git@github.com:wzzheng/GPD.git && cd GPD
sh ./script/setup_env.sh
```
