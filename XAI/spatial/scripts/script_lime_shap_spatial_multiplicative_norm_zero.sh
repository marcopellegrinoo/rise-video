#!/bin/bash

#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00                 # time limits: here 1 hour
#SBATCH --mem=80000                     # total memory per node requested in GB (optional)
#SBATCH --error=/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial/logs/lime_shap_spatial_multiplicative_norm_zero.err               # standard error file
#SBATCH --output=/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial/logs/lime_shap_spatial_multiplicative_norm_zero.out              # standard output file
#SBATCH --account=try25_pellegrino      # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --qos=normal                    # quality of service (optional)

export PYTHONUNBUFFERED=TRUE

module purge

source /leonardo_work/try25_pellegrino/Water_Resources/env/bin/activate
cd /leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial/scripts

TIMESTAMP_VAR=$(date +"%Y%m%d%H%M%S")
RESULT_PATH="/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial/results/paper_results"
DIR_NAME="01_spatial_lime_shap_"$TIMESTAMP_VAR
mkdir $RESULT_PATH/$DIR_NAME

python lime_shap_spatial_multiplicative_norm_zero.py $RESULT_PATH/$DIR_NAME