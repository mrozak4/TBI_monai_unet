#!/bin/bash
#SBATCH --account=rrg-bojana_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=124GB
#SBATCH --array=0-760
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/SAGE-%A_%a.out --array=0-760
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source ~/projects/def-gstanisz/rozakmat/torch_geom/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/Train_Graph_CNN.py -c ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/hyperparameter_pickle_files/GraphSAGE$SLURM_ARRAY_TASK_ID.pickle
