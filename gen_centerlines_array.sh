#!/bin/bash
#SBATCH --account=def-mgoubran
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --mem=8GB
#SBATCH --array=0-1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/gen_cent-%A_%a.out --array=0-1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source ~/projects/def-bojana/rozakmat/monai3.8/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/gen_centerlines.py -c $SLURM_ARRAY_TASK_ID
