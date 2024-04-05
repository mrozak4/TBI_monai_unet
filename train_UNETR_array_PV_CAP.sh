#!/bin/bash
#SBATCH --account=rrg-bojana_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=240GB
#SBATCH --array=434-442
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/unet_parallel_PV_CAP-%A_%a.out --array=434-442
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source /home/rozakmat/projects/def-bojana/rozakmat/monai3.8/bin/activate
python /home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/train_UNETR_optimize_hyperparameters_PV_CAP.py -c /home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/hyperparameter_pickle_files/parameters$SLURM_ARRAY_TASK_ID.pickle

