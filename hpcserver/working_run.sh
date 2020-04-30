#!/bin/sh
#SBATCH -p gpup100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=11:30:0
#SBATCH --workdir=/home-3/mfigdor1@jhu.edu/scratch/dldo/learn2branch
#SBATCH --output=/home-3/mfigdor1@jhu.edu/scratch/dldo/learn2branch/logs/train1.slurm.%j.out
#SBATCH --error=/home-3/mfigdor1@jhu.edu/scratch/dldo/learn2branch/logs/train1.slurm.%j.err
#SBATCH --job-name=train1
#SBATCH --mail-type=END
#SBATCH --mail-user=mfigdor1@jhu.edu

ml python/3.6
ml cmake
ml cuda/9.0

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

#cd ..
SEED=210
PROBLEM='setcover'

echo $SEED;
echo $PROBLEM;

python ./03_train_gcnn.py $PROBLEM --seed $SEED --sourcedir $DATADIR
# 3. run signularity image w/ python script
#singularity exec -B /scratch/ --nv ./hpcserver/sciptflow.sif python3.6 ./03_train_gcnn.py  $PROBLEM --seed $SEED --sourcedir $DATADIR

exit
