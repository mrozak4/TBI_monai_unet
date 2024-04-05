#!/bin/bash
#SBATCH --account=def-bojana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=498GB
#SBATCH --time=6:00:00
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module load anaconda3
source activate jupyter_env
python /scratch/b/bojana/rozakmat/TBI_monai_UNET/train_unet_optimize_hyperparameters.py -c /scratch/b/bojana/rozakmat/TBI_monai_UNET/hyperparameter_pickle_files/parameters81.pickle
