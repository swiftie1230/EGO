#!/bin/bash
#SBATCH --job-name  BASE-generation
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       45G
#SBATCH --gpus      0
#SBATCH --constraint=ada

source ~/.bashrc
conda activate RoleConflict

export OPENROUTER_API_KEY="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"

BACKEND=openrouter
LIMIT=200

MODEL=mistralai/mixtral-8x7b-instruct   #meta-llama/llama-3.1-70b-instruct
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')   # google/gemma-2b-it -> google_gemma-2b-it

OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs
DATADIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data

python src/BaseGeneration.py \
  --input_file $DATADIR/triage_allocation.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --or_model $MODEL \
  --out_file $OUTDIR/triage/triage_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 4 \
  --max_new_tokens 50 \
  --temperature 0.0

python src/BaseGeneration.py \
  --input_file $DATADIR/roleconflict_allocation.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --or_model $MODEL \
  --out_file $OUTDIR/roleconflict/roleconflict_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 2 \
  --max_new_tokens 50 \
  --temperature 0.0

python src/BaseGeneration.py \
  --input_file $DATADIR/ggb_skeleton.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --or_model $MODEL \
  --out_file $OUTDIR/moralDilemma/ggb_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 4 \
  --max_new_tokens 50 \
  --temperature 0.0

python src/BaseGeneration.py \
  --input_file $DATADIR/unibench_skeleton.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --or_model $MODEL \
  --out_file $OUTDIR/moralDilemma/unibench_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 4 \
  --max_new_tokens 50 \
  --temperature 0.0