#!/bin/bash
#SBATCH --job-name  paraphrase-BT
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate FRAMING

PYTHONNOUSERSITE=1 python src/paraphrase_bt.py \
  --input_dir "/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data" \
  --output_dir "/home/swiftie1230/EGO/FRAMING/FramingSensitivity/paraphrase/data" \
  --mode expand \
  --n 3 \
  --device cuda \
  --limit 400