#!/bin/bash
#SBATCH --account=def-cdemore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --array=192-240
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/GCN-%A_%a.out --array=192-240
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source ~/projects/def-gstanisz/rozakmat/torch_geom/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/Train_Graph_GCN.py -c ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/hyperparameter_pickle_files/GCN$SLURM_ARRAY_TASK_ID.pickle
