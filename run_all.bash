#!/bin/bash
../random-utils/run_sbatch.py 'EntNet.py --tasks=1 --train --n_tries=2 --output="./script_results/task_1.out"' --gpu=1
