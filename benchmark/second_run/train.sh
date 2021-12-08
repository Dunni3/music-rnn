#!/bin/bash
#SBATCH --job-name=bm_longer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=dept_cpu
#SBATCH --mem=16gb
#SBATCH --time=672:00:00
#SBATCH --qos=normal
#SBATCH -o job.stdout
#SBATCH -e job.stderr
#SBATCH --dependency=singleton
#SBATCH --mail-type=fail
#SBATCH --mail-user=ian.dunn@pitt.edu

source ~/.bashrc
cd $SLURM_SUBMIT_DIR
module load anaconda
conda activate mg
echo Running on $(hostname)
python ../train.py ./config.json

