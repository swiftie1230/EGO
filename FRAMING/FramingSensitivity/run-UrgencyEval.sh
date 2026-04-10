#!/bin/bash
#SBATCH --job-name  BASE-urgencyeval
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       45G
#SBATCH --gpus      0
#SBATCH --constraint=ada

conda activate RoleConflict

LIMIT=200

MODEL=mistralai/mixtral-8x7b-instruct
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')   # google/gemma-2b-it -> google_gemma-2b-it

OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs
DATADIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data

python src/UrgencyEval.py \
  --skeleton_file $DATADIR/triage_allocation.jsonl \
  --pred_file $OUTDIR/triage/triage_${MODEL_TAG}_preds_${LIMIT}.jsonl \

python src/UrgencyEval.py \
  --skeleton_file $DATADIR/roleconflict_allocation.jsonl \
  --pred_file $OUTDIR/roleconflict/roleconflict_${MODEL_TAG}_preds_${LIMIT}.jsonl \