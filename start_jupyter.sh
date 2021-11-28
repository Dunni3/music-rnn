#!/bin/bash

#SBATCH --job-name=rnn-jupyter

#SBATCH -t 8:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH -x g005
#SBATCH --mem=64gb

#SBATCH --output jupyter.stdout
#SBATCH -e jupyter.stderr
## get tunneling info

XDG_RUNTIME_DIR=""

ipnport=$(shuf -i8000-9999 -n1)

ipnip=$(hostname -i)

token=$(xxd -l 32 -c 32 -p < /dev/random)

## print tunneling instructions to sbatch.stdout

echo -e "

Copy/Paste this in your local terminal to ssh tunnel with remote

-----------------------------------------------------------------

ssh -N -L $ipnport:$ipnip:$ipnport $USER@cluster.csb.pitt.edu

-----------------------------------------------------------------

Then open a browser on your local machine to the following address

------------------------------------------------------------------

http://localhost:$ipnport?token=$token

------------------------------------------------------------------

"

## start an ipcluster instance and launch jupyter server
source ~/.bashrc
module load cuda/10.2
module load anaconda
conda activate mg

jupyter lab --NotebookApp.iopub_data_rate_limit=100000000000000 \
                 --port=$ipnport --ip=$ipnip \
                 --NotebookApp.password='' \
                 --NotebookApp.token="$token" --no-browser
