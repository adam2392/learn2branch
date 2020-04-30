#!/bin/bash
#SBATCH -p gpup100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=11:30:0
#SBATCH --workdir=/home-1/ali39@jhu.edu/code/learn2branch
#SBATCH --output=/home-1/ali39@jhu.edu/code/learn2branch/logs/train.slurm.%j.out
#SBATCH --error=/home-1/ali39@jhu.edu/code/learn2branch/logs/train.slurm.%j.err
#SBATCH --job-name=train1
#SBATCH --mail-type=END
#SBATCH --mail-user=ali39@jhu.edu

# helper cli commands to run interactive shell - MARCC
#interact -p debug -n 4 -t 1:0:0
#interact -t 3:0:0 -p gpuk80 -g 1 -N 1 -n 6

# load in CUDA/Singularity
module load cuda/9.2           # also locates matching $CUDA_DRIVER location
module load singularity/3.5

# load in Anaconda and conda environment
#module restore conda

# double check loaded modules
ml

######################################################################
# Use Singularity
######################################################################
# 1. (optional) pull tensorflow image if needed
# singularity pull --name tensorflow.simg shub://marcc-hpc/tensorflow
# singularity pull --arch amd64 ./sciptflow.sif library://adam2392/default/scip_and_deeplearning:latest

# 2. redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

# 3. run signularity image w/ python script
singularity exec --nv ./sciptflow.sif python3.6 ../04_test.py <params>

exit