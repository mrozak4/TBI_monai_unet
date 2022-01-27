#!/bin/bash
#SBATCH --account=def-bojana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=498GB
#SBATCH --time=6:00:00
#SBATCH --array=1-192
#SBATCH --out=/home/rozakmat/projects/def-bojana/rozakmat/TBI_monai_UNET/logs/unet_parallel-%A_%a.out --array=1-192
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module load python/3.7
source /home/rozakmat/projects/def-bojana/rozakmat/monai/bin/activate
python /home/rozakmat/projects/def-bojana/rozakmat/TBI_monai_UNET/train_unet_optimize_hyperparameters.py -c /home/rozakmat/projects/def-bojana/rozakmat/TBI_monai_UNET/hyperparameter_pickle_files/parameters$SLURM_ARRAY_TASK_ID.pickle
