#!/bin/bash
#SBATCH --account=rrg-bojana_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --mem=498GB
#SBATCH --array=434-441
#SBATCH --cpus-per-task=48
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source ~/projects/def-bojana/rozakmat/monai3.8/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/train_UNETR_optimize_hyperparameters.py -c ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/hyperparameter_pickle_files/parameters$SLURM_ARRAY_TASK_ID.pickle

