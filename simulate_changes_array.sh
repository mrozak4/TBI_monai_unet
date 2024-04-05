#!/bin/bash
#SBATCH --account=def-mgoubran
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=30GB
#SBATCH --array=0-300
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/sim_dilation-%A_%a.out --array=0-300
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --dependency=afterok:17031691

source ~/projects/def-bojana/rozakmat/monai3.8/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/simulate_changes_noise.py -c $SLURM_ARRAY_TASK_ID
