#!/bin/bash
#SBATCH --job-name=learn2branch
#SBATCH --time=00:10:0
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#interact -p debug -n 4 -t 1:0:0

module restore conda
ml

# echo which python we are using
echo 'PYTHON IS'
echo $(which python)

# actual bash commands to submit the job

exit