#!/bin/bash
#SBATCH --job-name  SCOTUS
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       45G
#SBATCH --gpus      1
#SBATCH --constraint=ada

conda activate RoleConflict

python src/SCOTUSSkeleton.py