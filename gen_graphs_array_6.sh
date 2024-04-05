#!/bin/bash
#SBATCH --account=def-bojana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6GB
#SBATCH --time=3:00:00
#SBATCH --array=0-40
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/gen_graphs-%A_%a.out --array=0-40
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --dependency=afterok:17035274

source ~/projects/def-bojana/rozakmat/monai3.8/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/gen_graphs.py -c $SLURM_ARRAY_TASK_ID
