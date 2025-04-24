#!/bin/bash
#BSUB -J proximal_gradient_backtracking
#BSUB -q hpc
#BSUB -W 60
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=512MB]"
#BSUB -o proximal_gradient_backtracking_%J.out
#BSUB -e proximal_gradient_backtracking_%J.err

source /zhome/b0/3/214044/miniconda3/etc/profile.d/conda.sh
conda activate ds_optim

time python test_new.py
