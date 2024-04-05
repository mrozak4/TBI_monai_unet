#!/bin/bash
#SBATCH --account=rrg-bojana_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=498GB
#SBATCH --time=3:00:00
#SBATCH --array=0-2
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/predict_ood-%A_%a.out --array=0-2
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source /home/rozakmat/projects/def-bojana/rozakmat/monai3.8/bin/activate
python /home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/predict_OOD.py
