#!/bin/bash

#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00                 # time limits: here 1 hour
#SBATCH --mem=80000                                                                                                     # total memory per node requested in GB (optional)
#SBATCH --error=/leonardo_scratch/fast/try25_pellegrino/results/logs/rise_st_additive_gaussian_single_noise.err         # standard error file
#SBATCH --output=/leonardo_scratch/fast/try25_pellegrino/results/logs/rise_st_additive_gaussian_single_noise.out        # standard output file
#SBATCH --account=try25_pellegrino                                                                                      # account name
#SBATCH --partition=boost_usr_prod                                                                                      # partition name
#SBATCH --qos=normal                                                                                                    # quality of service (optional)

export PYTHONUNBUFFERED=TRUE

source $WORK/Water_Resources/env/bin/activate
cd $WORK/Water_Resources/rise-video/XAI/temporal/scripts

python rise_st_additive_gaussian_single_noise.py
