#!/bin/bash
../random-utils/run_sbatch.py 'EntNet.py --tasks=1 --train --n_tries=2' --gpu=1
