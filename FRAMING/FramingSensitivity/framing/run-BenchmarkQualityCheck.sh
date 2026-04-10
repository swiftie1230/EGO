#!/bin/bash
#SBATCH --job-name=BenchmarkQualityCheck
#SBATCH --time=100:00:00
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH --gpus=0
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate RoleConflict

python src/BenchmarkQualityCheck.py \
  --value-tinted-path /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/value_tinted_narration/output/v4 \
  --experiential-path /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/experiential_framing/output/v7/vividness \
  --temporal-path /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v4 \
  --units-output /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing_qc/outputs/eval_units.jsonl \
  --judged-output /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing_qc/outputs/judged.jsonl \
  --summary-output /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing_qc/outputs/summary.json \
  --human-sample-output /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing_qc/outputs/human_sample.jsonl \
  --model openai/gpt-4o-mini-2024-07-18 \
  --limit-per-framing 200