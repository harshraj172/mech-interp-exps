#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p boost_usr_prod
#SBATCH --time 8:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1          # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=gemma_ft
#SBATCH --output=slurm_out/gemma_ft-%j-%t.out


source ~/miniconda3/bin/activate

srun python -m gemma_ft_lora 