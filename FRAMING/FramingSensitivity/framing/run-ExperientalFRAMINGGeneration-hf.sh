#!/bin/bash
#SBATCH --job-name  FRAMING-generation_llama
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       70G
#SBATCH --gpus      1
#SBATCH --constraint=ada

source ~/.bashrc
conda activate RoleConflict

# -----------------------------
# SETTINGS
# -----------------------------
BACKEND=hf
#LIMIT=4

MODEL=meta-llama/Llama-3.1-8B-Instruct
#MODEL=mistralai/Mistral-7B-Instruct-v0.3 
#MODEL=Qwen/Qwen2.5-7B-Instruct 
#MODEL=google/gemma-2-9b-it
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')

PRED_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/v6
FRAMING_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/experiential_framing/output/v7/vividness
OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outputs/experiential_framing

# ==========================================
# TRIAGE
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_DIR/triage \
  --model_tag $MODEL_TAG \
  --framing_path $FRAMING_DIR/triage_allocation__Qwen_Qwen2.5-7B-Instruct__limit400.jsonl \
  --out_path $OUTDIR/triage/v1 \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
#  --limit $LIMIT

# ==========================================
# ROLECONFLICT
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_DIR/roleconflict \
  --model_tag $MODEL_TAG \
  --framing_path $FRAMING_DIR/roleconflict_allocation__Qwen_Qwen2.5-7B-Instruct__limit400.jsonl \
  --out_path $OUTDIR/roleconflict/v1 \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
#  --limit $LIMIT

# ==========================================
# GGB
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_DIR/moralDilemma \
  --model_tag $MODEL_TAG \
  --dataset_prefix ggb \
  --framing_path $FRAMING_DIR/ggb_skeleton__Qwen_Qwen2.5-7B-Instruct__limit400.jsonl \
  --out_path $OUTDIR/moralDilemma/v1 \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
#  --limit $LIMIT

# ==========================================
# UNIBENCH
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_DIR/moralDilemma \
  --model_tag $MODEL_TAG \
  --dataset_prefix unibench \
  --framing_path $FRAMING_DIR/unibench_skeleton__Qwen_Qwen2.5-7B-Instruct__limit400.jsonl \
  --out_path $OUTDIR/moralDilemma/v1 \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
#  --limit $LIMIT
  
  
# ==========================================
# SUPER-SCOTUS (legal_decision)
# ==========================================
python src/ExperientalFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_DIR/SCOTUS \
  --model_tag $MODEL_TAG \
  --framing_path $FRAMING_DIR/SCOTUS_skeleton__Qwen_Qwen2.5-7B-Instruct__limit400.jsonl \
  --out_path $OUTDIR/SCOTUS/v1 \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
#  --limit $LIMIT

# ==========================================
# Medical Triage Alignment
# ==========================================
#python src/ExperientalFramingGeneration.py \
#  --backend $BACKEND \
#  --model_id $MODEL \
#  --pred_dir $PRED_DIR/medical_triage \
#  --model_tag $MODEL_TAG \
#  --framing_path $FRAMING_DIR/medical_triage_alignment_skeleton__Qwen_Qwen2.5-7B-Instruct__limit400.jsonl \
#  --out_path $OUTDIR/medical_triage/v1 \
#  --batch_size 4 \
#  --max_new_tokens 50 \
#  --temperature 0.0
#  --limit $LIMIT

echo "===== HF Counter Framing DONE ====="

