#!/bin/bash
#SBATCH --account=def-cdemore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=249GB
#SBATCH --array=0-80
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=matthew.rozak@mail.utoronto.ca
#SBATCH --out=/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/logs/grad_bound-%A_%a.out --array=0-80
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source ~/projects/def-bojana/rozakmat/monai3.8/bin/activate
python ~/projects/rrg-bojana/rozakmat/TBI_monai_UNET/graph_grad_boundary_calc.py -c $SLURM_ARRAY_TASK_ID
