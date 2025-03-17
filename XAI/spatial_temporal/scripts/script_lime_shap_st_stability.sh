#!/bin/bash

#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00                 # time limits: here 1 hour
#SBATCH --mem=40000                     # total memory per node requested in GB (optional)
#SBATCH --error=/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/logs/05_marco_st_lime_shap_stability.err               # standard error file
#SBATCH --output=/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/logs/05_marco_st_lime_shap_stability.out              # standard output file
#SBATCH --account=IscrC_DL4EO           # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --qos=normal                    # quality of service (optional)

export PYTHONUNBUFFERED=TRUE

module purge

source /leonardo_work/try25_pellegrino/Water_Resources/env/bin/activate
cd /leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/scripts

TIMESTAMP_VAR=$(date +"%Y%m%d%H%M%S")
RESULT_PATH="/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/results"
DIR_NAME="05_marco_st_lime_shap_stability"$TIMESTAMP_VAR

mkdir $RESULT_PATH/$DIR_NAME

python lime_shap_st_stability_5.py $RESULT_PATH/$DIR_NAME