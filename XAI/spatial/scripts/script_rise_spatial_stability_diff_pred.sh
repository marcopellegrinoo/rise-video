#!/bin/bash

#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00                 # time limits: here 1 hour
#SBATCH --error=/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial/logs/rise_spatial_stability_diff_pred.err               # standard error file
#SBATCH --output=/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial/logs/rise_spatial_stability_diff_pred.out              # standard output file
#SBATCH --account=try25_pellegrino      # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --qos=normal                    # quality of service (optional)

export PYTHONUNBUFFERED=TRUE

module purge

source $WORK/Water_Resources/env/bin/activate
cd $WORK/Water_Resources/rise-video/XAI/spatial/scripts

python rise_spatial_stability_diff_pred.py