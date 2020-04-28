#!/bin/bash
#SBATCH --partition=gpup100
#SBATCH –gres=gpu:1
#SBATCH --workdir=/home-1/ali39@jhu.edu/code/
#SBATCH --output=train.slurm.%j.out
#SBATCH --error=train.slurm.%j.err
#SBATCH --job-name=train
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mail-type=END
#SBATCH --mail-user=ali39@jhu.edu

#interact -p debug -n 4 -t 1:0:0
#interact -t 3:0:0 -p gpuk80 -g 1 -N 1 -n 6

# load in CUDA/Singularity
module load cuda/9.2           # also locates matching $CUDA_DRIVER location
module load singularity/3.5

# load in Anaconda and conda environment
#module restore conda

# activate  conda environment and double check loaded modules
ml

# echo which python we are using
echo 'PYTHON IS'
echo $(which python)

DATADIR="$HOME/data/learn2branch/"

# actual bash commands to submit the job
######################################################################
# Use Singularity
######################################################################
# 1. (optional) pull tensorflow image if needed
# singularity pull --name tensorflow.simg shub://marcc-hpc/tensorflow
# singularity pull --arch amd64 ./sciptflow.sif library://adam2392/default/scip_and_deeplearning:latest

# 2. redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

cd ..

# 3. run signularity image w/ python script
singularity exec --nv ./sciptflow.sif python3.6 ../03_train_gcnn.py cauctions --sourcedir DATADIR

exit