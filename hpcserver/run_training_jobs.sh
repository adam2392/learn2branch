#!/bin/bash
#SBATCH --job-name=learn2branch
#SBATCH --time=00:10:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH -o tensorflow-%j.out
#SBATCH --partition=debug

####SBATCH --partition=gpuk80
####SBATCH --gres=gpu:1
#interact -p debug -n 4 -t 1:0:0

# load in CUDA/Singularity
module load cuda           # also locates matching $CUDA_DRIVER location
module load singularity

# load in Anaconda and conda environment
module restore conda

# activate  conda environment and double check loaded modules
ml
source activate learn2branch
conda activate learn2branch

# echo which python we are using
echo 'PYTHON IS'
echo $(which python)

# actual bash commands to submit the job

######################################################################
# Use Singularity
######################################################################
# pull tensorflow image
singularity pull --name tensorflow.simg shub://marcc-hpc/tensorflow
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER
# run signularity image w/ python script
singularity exec --nv ./tensorflow.simg python softmax_regression.py

######################################################################
# just submit via python?
######################################################################
python ./run_training.py


exit