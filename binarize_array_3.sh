#!/bin/bash
#SBATCH --account=rrg-bojana_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --begin=now+6hour
#SBATCH --mem=5GB
#SBATCH --array=0-30
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/binarize-%A_%a.out --array=0-30
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE



source ~/projects/def-bojana/rozakmat/monai3.8/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/binarize.py -c $SLURM_ARRAY_TASK_ID
