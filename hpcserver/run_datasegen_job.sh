#!/bin/sh
#SBATCH --partition=shared
#SBATCH --workdir=/home-1/ali39@jhu.edu/code/
#SBATCH --output=tsp.slurm.%j.out
#SBATCH --error=tsp.slurm.%j.err
#SBATCH --job-name=tsp_data
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mail-type=END
#SBATCH --mail-user=ali39@jhu.edu

module load cuda/9.2
module load singularity/3.5

# actual bash commands to submit the job
######################################################################
# Use Singularity
######################################################################
# 1. (optional) pull tensorflow image if needed
# singularity pull --name tensorflow.simg shub://marcc-hpc/tensorflow
# singularity pull --arch amd64 ./sciptflow.sif library://adam2392/default/scip_and_deeplearning:latest

# 2. redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

# 3. run signularity image w/ python script
singularity exec ./sciptflow.sif python ../02_generate_dataset.py --<params>

exit