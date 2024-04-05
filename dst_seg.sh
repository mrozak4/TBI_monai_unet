#!/bin/bash
#SBATCH --account=def-bojana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=249GB
#SBATCH --time=12:00:00
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source /home/rozakmat/projects/def-bojana/rozakmat/monai3.8/bin/activate
python /home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/calculate_dst_seg_matt.py
