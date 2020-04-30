#!/bin/sh
#SBATCH -p gpup100
#SBATCH –-gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=11:30:0
#SBATCH --workdir=/home-1/ali39@jhu.edu/code/
#SBATCH --output=/home-1/ali39@jhu.edu/code/learn2branch/logs/train.slurm.%j.out
#SBATCH --error=/home-1/ali39@jhu.edu/code/learn2branch/logs/train.slurm.%j.err
#SBATCH --job-name=train
#SBATCH --mail-type=END
#SBATCH --mail-user=ali39@jhu.edu

#interact -p debug -n 4 -t 1:0:0
#interact -t 11:30:0 -p gpup100 -g 1 -N 1 -n 6

ml python/3.6
ml cmake
ml cuda/9.0
ml singularity/3.5

# load in CUDA/Singularity
# ml cuda/10.1
# https://marcc-hpc.github.io/esc/common/tensorflow-latest
#ml gcc/6.4.0
#ml cuda/10.0          # also locates matching $CUDA_DRIVER location
#module load singularity/3.5

# load in Anaconda and conda environment
#module restore conda

# activate  conda environment and double check loaded modules
ml

# echo which python we are using
echo 'PYTHON IS'
echo $(which python)
echo $CUDA_VISIBLE_DEVICES

#source ../.venv/bin/activate
export SCIPOPTDIR="$HOME/code/scip"
DATADIR="$HOME/data/learn2branch/"

#LD_LIBRARY_PATH="$HOME/code/learn2branch/.venv/lib64/":$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/lib:/usr/lib:$LD_LIBRARY_PATH
#gcc --version
#echo $(which python)

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
SEED=0
PROBLEM='cauctions'

echo $SEED;
echo $PROBLEM;

python ./03_train_gcnn.py $PROBLEM --seed $SEED --sourcedir $DATADIR
# 3. run signularity image w/ python script
#singularity exec -B /scratch/ --nv ./hpcserver/sciptflow.sif python3.6 ./03_train_gcnn.py  $PROBLEM --seed $SEED --sourcedir $DATADIR

exit
