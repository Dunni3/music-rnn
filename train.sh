#!/bin/bash
#SBATCH --job-name=insertjobname
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH -x g005
#SBATCH --mem=64gb
#SBATCH --time=672:00:00
#SBATCH --qos=normal
#SBATCH -o %J.stdout
#SBATCH -e %J.stderr
#SBATCH --dependency=singleton
#SBATCH --mail-type=fail
#SBATCH --mail-user=ian.dunn@pitt.edu
#SBATCH -C M12

source ~/.bashrc
cd $SLURM_SUBMIT_DIR
module load cuda/10.2
module load anaconda
conda activate mg
echo Running on $(hostname)
python /net/dali/home/mscbio/icd3/myligan/train.py ./train.config

