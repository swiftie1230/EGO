#!/bin/bash
#SBATCH --job-name  FRAMING-generation
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

MODEL_TAG=Qwen_Qwen2.5-7B-Instruct
OR_MODEL=meta-llama/llama-3.1-70b-instruct

PRED_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs
FRAMING_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/experiential_framing/output/v3/vividness
OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outputs

# ==========================================
# TRIAGE
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --or_model $OR_MODEL \
  --pred_dir $PRED_DIR/triage \
  --model_tag $MODEL_TAG \
  --framing_path $FRAMING_DIR/triage_allocation__Qwen_Qwen2.5-7B-Instruct.jsonl \
  --out_path $OUTDIR/triage/ \
  --max_new_tokens 50 \
  --temperature 0.0 \
  --limit $LIMIT

# ==========================================
# ROLECONFLICT
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --or_model $OR_MODEL \
  --pred_dir $PRED_DIR/roleconflict \
  --model_tag $MODEL_TAG \
  --framing_path $FRAMING_DIR/roleconflict_allocation__Qwen_Qwen2.5-7B-Instruct.jsonl \
  --out_path $OUTDIR/roleconflict/ \
  --max_new_tokens 50 \
  --temperature 0.0 \
  --limit $LIMIT

# ==========================================
# GGB
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --or_model $OR_MODEL \
  --pred_dir $PRED_DIR/moralDilemma \
  --model_tag $MODEL_TAG \
  --dataset_prefix ggb \
  --framing_path $FRAMING_DIR/ggb_skeleton__Qwen_Qwen2.5-7B-Instruct.jsonl \
  --out_path $OUTDIR/moralDilemma/ \
  --max_new_tokens 50 \
  --temperature 0.0 \
  --limit $LIMIT

# ==========================================
# UNIBENCH
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --or_model $OR_MODEL \
  --pred_dir $PRED_DIR/triage \
  --model_tag $MODEL_TAG \
  --dataset_prefix unibench \
  --framing_path $FRAMING_DIR/unibench_skeleton__Qwen_Qwen2.5-7B-Instruct.jsonl \
  --out_path $OUTDIR/moralDilemma/ \
  --max_new_tokens 50 \
  --temperature 0.0 \
  --limit $LIMIT

echo "===== OpenRouter Counter Framing DONE ====="
