#!/bin/bash
#SBATCH --job-name  FRAMING-generation-mistral
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx

source ~/.bashrc
conda activate RoleConflict

# -----------------------------
# SETTINGS
# -----------------------------
BACKEND=hf
LIMIT=200

MODEL=mistralai/Mistral-7B-Instruct-v0.3 #meta-llama/Llama-3.1-8B-Instruct #Qwen/Qwen2.5-7B-Instruct #google/gemma-2-9b-it
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')

PRED_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs
FRAMING_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/experiential_framing/output/v3/vividness
OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outputs/experiential_framing

# ==========================================
# TRIAGE
# ==========================================
#python src/ExperientalFramingGeneration.py \
#  --backend $BACKEND \
#  --model_id $MODEL \
#  --pred_dir /home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/triage \
#  --model_tag $MODEL_TAG \
#  --framing_path $FRAMING_DIR/triage_allocation__Qwen_Qwen2.5-7B-Instruct.jsonl \
#  --out_path $OUTDIR/triage/ \
#  --batch_size 4 \
#  --max_new_tokens 50 \
#  --temperature 0.0
#  --limit $LIMIT

# ==========================================
# ROLECONFLICT
# ==========================================
#python src/ExperientalFramingGeneration.py \
#  --backend $BACKEND \
#  --model_id $MODEL \
#  --pred_dir /home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/roleconflict \
#  --model_tag $MODEL_TAG \
#  --framing_path $FRAMING_DIR/roleconflict_allocation__Qwen_Qwen2.5-7B-Instruct.jsonl \
#  --out_path $OUTDIR/roleconflict/ \
#  --batch_size 4 \
#  --max_new_tokens 50 \
#  --temperature 0.0
#  --limit $LIMIT

# ==========================================
# GGB
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir /home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/moralDilemma \
  --model_tag $MODEL_TAG \
  --dataset_prefix ggb \
  --framing_path $FRAMING_DIR/ggb_skeleton__Qwen_Qwen2.5-7B-Instruct.jsonl \
  --out_path $OUTDIR/moralDilemma/v2 \
  --batch_size 4 \
  --max_new_tokens 50 \
  --temperature 0.0
  --limit $LIMIT

# ==========================================
# UNIBENCH
# ==========================================
#python src/ExperientalFramingGeneration.py \
#  --backend $BACKEND \
#  --model_id $MODEL \
#  --pred_dir /home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/moralDilemma \
#  --model_tag $MODEL_TAG \
#  --dataset_prefix unibench \
#  --framing_path $FRAMING_DIR/unibench_skeleton__Qwen_Qwen2.5-7B-Instruct.jsonl \
#  --out_path $OUTDIR/moralDilemma/ \
#  --batch_size 4 \
#  --max_new_tokens 50 \
#  --temperature 0.0
#  --limit $LIMIT

echo "===== HF Counter Framing DONE ====="

