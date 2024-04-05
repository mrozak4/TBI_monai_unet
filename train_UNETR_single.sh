#!/bin/bash
#SBATCH --account=def-bojana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH -p compute_full_node
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module load anaconda3
source activate jupyter_env
python /scratch/b/bojana/rozakmat/TBI_monai_UNET/train_UNETR_optimize_hyperparameters.py -c /scratch/b/bojana/rozakmat/TBI_monai_UNET/hyperparameter_pickle_files/parameters300.pickle

