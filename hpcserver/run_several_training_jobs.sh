#!/bin/sh

problems=( setcover cauctions indset tsp )
for i in "${problems[@]}"
do
	sbatch run_training_jobs.sh $i
done
