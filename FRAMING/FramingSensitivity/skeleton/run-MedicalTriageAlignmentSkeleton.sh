#!/bin/bash
#SBATCH --job-name  MedicalTriageAlignment
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       45G
#SBATCH --gpus      0
#SBATCH --constraint=rtx

conda activate RoleConflict

python src/MedicalTriageAlignmentSkeleton.py