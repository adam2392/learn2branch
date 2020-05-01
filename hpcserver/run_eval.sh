#!/bin/sh
#SBATCH -p gpuk80
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=5:00:0
#SBATCH --workdir=/home-1/ali39@jhu.edu/code/learn2branch
#SBATCH --output=/home-1/ali39@jhu.edu/code/learn2branch/logs/eval.slurm.%j.out
#SBATCH --error=/home-1/ali39@jhu.edu/code/learn2branch/logs/eval.slurm.%j.err
#SBATCH --job-name=eval1

ml python/3.6
ml cmake
ml cuda/9.0
ml singularity/3.5

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
#singularity pull --arch amd64 ./sciptflow.sif library://adam2392/default/scip_and_deeplearning:latest

# 2. redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

#cd ..
#SEED='210,211,212'
SEED='0,1,2,22,202,206,211' # cauctions
PROBLEM='cauctions'

#SEED='12,209' # indset
#PROBLEM='indset'

#SEED='210' # set cover
#PROBLEM='setcover'

#SEED='207' # tsp
#PROBLEM='tsp'

echo $SEED;
echo $PROBLEM;

# 3. run signularity image w/ python script
singularity exec -B /scratch/ --nv ./hpcserver/sciptflow.sif python3.6 ./05_evaluate.py $PROBLEM --seed $SEED --sourcedir $DATADIR

exit
