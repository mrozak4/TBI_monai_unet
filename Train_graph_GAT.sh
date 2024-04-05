#!/bin/bash
#SBATCH --account=def-bojana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --array=0-480
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/GAT-%A_%a.out --array=0-480
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source ~/projects/def-gstanisz/rozakmat/torch_geom/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/Train_Graph_GAT.py -c ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/hyperparameter_pickle_files/GAT$SLURM_ARRAY_TASK_ID.pickle
