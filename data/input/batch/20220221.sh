#!/bin/bash
#SBATCH -a 0-1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-08:00
#SBATCH -o /pool001/jacobwac/wildfires/data/output/20220221.out
#SBATCH -e /pool001/jacobwac/wildfires/data/output/20220221.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jacobwac@mit.edu

module load anaconda3/4.4.0
source activate bare

python cnn.py 20220221 $SLURM_ARRAY_TASK_ID
