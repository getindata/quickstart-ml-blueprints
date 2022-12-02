# Graph Neural Network-based recommender system for sequential data

## Overview


## Data
Easiest way to download data is by creating Kaggle account and downloading data from [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data). You can also download tabular data from [Google Drive](https://drive.google.com/drive/folders/1sdV7_nGqC-MvE4GXwC12btCGDIZplHa4?usp=sharing) (and sample of images).

## How to prepare GPU environment with Conda and Poetry

First install miniconda and Poetry for you OS version. Next create conda-lock file with requirements:
```
conda create -n temp -c conda-forge mamba conda-lock poetry='1.*' python='3.8.12' && conda clean -afy
conda activate temp
conda-lock -k explicit --conda mamba
poetry add --lock torch=1.12.1 torchaudio=0.12.1 torchvision=0.13.1 conda-lock
conda activate base
conda env remove -n temp
```

Create new conda environment and install poetry dependencies:

```
conda create --name recommender_gnn --file conda-linux-64.lock && conda clean -afy
conda activate recommender_gnn
poetry install
```

## How to setup pre-commit

To setup pre-commit hooks: 
```
pre-commit install
```

## How to run Kedro

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, look at the `.coveragerc` file.
