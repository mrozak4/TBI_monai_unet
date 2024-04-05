#!/bin/bash
#SBATCH --account=rrg-bojana_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source /home/rozakmat/projects/def-bojana/rozakmat/monai3.8/bin/activate
python /home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/predict_matt_1x.py 
