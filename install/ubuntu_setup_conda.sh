#!/bin/sh
#Create a bootstrap env 
conda create -p /tmp/bootstrap -c conda-forge mamba conda-lock poetry='1.*' python=`3.8.12`
conda activate /tmp/bootstrap
#Create Conda lock file(s) from environment.yml 
conda-lock -k explicit --conda mamba 
#Set up Poetry
poetry add --lock torch=1.12.1 torchaudio=0.12.1 torchvision=0.13.1
poetry add --lock conda-lock
#Remove the bootstrap env 
conda deactivate rm -rf /tmp/bootstrap
#New project environment
conda create --name gid_ml_framework --file conda-linux-64.lock
conda activate gid_ml_framework
poetry install
