#!/bin/sh

problems=( setcover cauctions indset tsp )
for ((i = 0; i < ${#problems[@]}; ++i)); do
    # bash arrays are 0-indexed
    position=$(( $i + 1 ))
    sbatch run_training_jobs_2_bulk.sh ${problems[$i]} 10$position
done
