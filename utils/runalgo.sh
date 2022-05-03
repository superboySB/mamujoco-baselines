#!/bin/bash

envs=( MMM2 corridor 3s_vs_5z)
#envs=(MMM2 corridor 3s_vs_5z)
#envs=(8m)

for e in "${envs[@]}"
do
   for i in {0..0}
   do
      python3.6 src/main.py --config=$1 --env-config=sc2 with env_args.map_name=$e seed=$2 
      echo "Running with $1 and $e for seed=$i"
   done
done
