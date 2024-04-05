#!/bin/bash
#SBATCH --account=def-bojana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=249GB
#SBATCH --time=0:30:00
#SBATCH --array=0-5
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/predict-%A_%a.out --array=0-5
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source /home/rozakmat/projects/def-bojana/rozakmat/monai3.8/bin/activate
python /home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/predict_matt_warped.py -c $SLURM_ARRAY_TASK_ID
