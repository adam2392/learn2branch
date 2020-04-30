#!/bin/sh

problems=( setcover cauctions indset tsp )
for ((i = 0; i < ${#problems[@]}; ++i)); do
    # bash arrays are 0-indexed
    #position=$(( $i + 1 ))
    sbatch working_run.sh ${problems[$i]} --seed 211
done
