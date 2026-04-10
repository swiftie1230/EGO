#!/bin/bash
#SBATCH --job-name  BASE-generation_mistral
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       70G
#SBATCH --gpus      1
#SBATCH --constraint=ada

source ~/.bashrc
conda activate RoleConflict

BACKEND=hf
LIMIT=400

#MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL=mistralai/Mistral-7B-Instruct-v0.3 
#MODEL=Qwen/Qwen2.5-7B-Instruct 
#MODEL=google/gemma-2-9b-it
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')   # google/gemma-2b-it -> google_gemma-2b-it

OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/v6
DATADIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data

python src/BaseGeneration.py \
  --input_file $DATADIR/triage_allocation.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --model_id $MODEL \
  --out_file $OUTDIR/triage/triage_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 4 \
  --max_new_tokens 1 \
  --temperature 0.7 \
  --decode_mode sample \
  --nbest 10 \
  --top_p 0.95

python src/BaseGeneration.py \
  --input_file $DATADIR/roleconflict_allocation.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --model_id $MODEL \
  --out_file $OUTDIR/roleconflict/roleconflict_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 2 \
  --max_new_tokens 1 \
  --temperature 0.7 \
  --decode_mode sample \
  --nbest 10 \
  --top_p 0.95

python src/BaseGeneration.py \
  --input_file $DATADIR/ggb_skeleton.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --model_id $MODEL \
  --out_file $OUTDIR/moralDilemma/ggb_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 4 \
  --max_new_tokens 1 \
  --temperature 0.7 \
  --decode_mode sample \
  --nbest 10 \
  --top_p 0.95

python src/BaseGeneration.py \
  --input_file $DATADIR/unibench_skeleton.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --model_id $MODEL \
  --out_file $OUTDIR/moralDilemma/unibench_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 4 \
  --max_new_tokens 1 \
  --temperature 0.7 \
  --decode_mode sample \
  --nbest 10 \
  --top_p 0.95
  
  
python src/BaseGeneration.py \
  --input_file $DATADIR/medical_triage_alignment_skeleton.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --model_id $MODEL \
  --out_file $OUTDIR/medical_triage/medical_triage_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 4 \
  --max_new_tokens 1 \
  --temperature 0.7 \
  --decode_mode sample \
  --nbest 10 \
  --top_p 0.95


python src/BaseGeneration.py \
  --input_file $DATADIR/SCOTUS_skeleton.jsonl \
  --limit $LIMIT \
  --backend $BACKEND \
  --model_id $MODEL \
  --out_file $OUTDIR/SCOTUS/SCOTUS_${MODEL_TAG}_preds_${LIMIT}.jsonl \
  --batch_size 4 \
  --max_new_tokens 1 \
  --temperature 0.7 \
  --decode_mode sample \
  --nbest 10 \
  --top_p 0.95