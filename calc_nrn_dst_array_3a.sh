#!/bin/bash
#SBATCH --account=def-bojana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=249GB
#SBATCH --array=0-30
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/nrn_dst-%A_%a.out --array=0-30
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --dependency=afterok:17035179

source ~/projects/def-bojana/rozakmat/monai3.8/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/calc_neuron_distance.py -c $SLURM_ARRAY_TASK_ID
