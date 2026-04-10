#!/bin/bash
#SBATCH --job-name=FRAMING-generation_qwen
#SBATCH --time=100:00:00
#SBATCH -c 4
#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --constraint=ada

source ~/.bashrc
conda activate RoleConflict

# -----------------------------
# SETTINGS
# -----------------------------
BACKEND=hf
LIMIT=400

#MODEL=meta-llama/Llama-3.1-8B-Instruct
#MODEL=mistralai/Mistral-7B-Instruct-v0.3 
MODEL=Qwen/Qwen2.5-7B-Instruct 
#MODEL=google/gemma-2-9b-it

MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')

PRED_ROOT=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/v6
FRAMING_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v4
OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outputs/outcome-oriented_framing/temporal_framing

# ==========================================
# TRIAGE
# ==========================================
python src/Outcome-OrientedFramingGeneration.py \
   --backend $BACKEND \
   --model_id $MODEL \
   --pred_dir $PRED_ROOT/triage \
   --model_tag $MODEL_TAG \
   --framing_path $FRAMING_DIR/triage_allocation_temporal_framing_Qwen_Qwen2.5-7B-Instruct_400.jsonl \
   --out_path $OUTDIR/triage/v1/ \
   --batch_size 4 \
   --max_new_tokens 50 \
   --decode_mode sample \
   --nbest 10 \
   --temperature 0.7 \
   --top_p 0.95 \
   --limit $LIMIT

# ==========================================
# ROLECONFLICT
# ==========================================
python src/Outcome-OrientedFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_ROOT/roleconflict \
  --model_tag $MODEL_TAG \
  --framing_path $FRAMING_DIR/roleconflict_allocation_temporal_framing_Qwen_Qwen2.5-7B-Instruct_400.jsonl \
  --out_path $OUTDIR/roleconflict/v1/ \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
  --limit $LIMIT
  
  
# ==========================================
# medical_triage
# ==========================================
#python src/Outcome-OrientedFramingGeneration.py \
#  --backend $BACKEND \
#  --model_id $MODEL \
#  --pred_dir $PRED_ROOT/medical_triage \
#  --model_tag $MODEL_TAG \
#  --framing_path $FRAMING_DIR/medical_triage_allocation_temporal_framing_Qwen_Qwen2.5-7B-Instruct_400.jsonl \
#  --out_path $OUTDIR/medical_triage/v1/ \
#  --batch_size 4 \
#  --max_new_tokens 50 \
#  --decode_mode sample \
#  --nbest 10 \
#  --temperature 0.7 \
#  --top_p 0.95 \
#  --limit $LIMIT
  
  
# ==========================================
# SCOTUS
# ==========================================
python src/Outcome-OrientedFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_ROOT/SCOTUS \
  --model_tag $MODEL_TAG \
  --framing_path $FRAMING_DIR/SCOTUS_skeleton_temporal_framing_Qwen_Qwen2.5-7B-Instruct_400.jsonl \
  --out_path $OUTDIR/SCOTUS/v1/ \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
  --limit $LIMIT

# ==========================================
# GGB
# ==========================================
python src/Outcome-OrientedFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_ROOT/moralDilemma \
  --model_tag $MODEL_TAG \
  --dataset_prefix ggb \
  --framing_path $FRAMING_DIR/ggb_skeleton_temporal_framing_Qwen_Qwen2.5-7B-Instruct_400.jsonl \
  --out_path $OUTDIR/moralDilemma/v1/ \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
  --limit $LIMIT

# ==========================================
# UNIBENCH
# ==========================================
python src/Outcome-OrientedFramingGeneration.py \
  --backend $BACKEND \
  --model_id $MODEL \
  --pred_dir $PRED_ROOT/moralDilemma \
  --model_tag $MODEL_TAG \
  --dataset_prefix unibench \
  --framing_path $FRAMING_DIR/unibench_skeleton_temporal_framing_Qwen_Qwen2.5-7B-Instruct_400.jsonl \
  --out_path $OUTDIR/moralDilemma/v1/ \
  --batch_size 4 \
  --max_new_tokens 50 \
  --decode_mode sample \
  --nbest 10 \
  --temperature 0.7 \
  --top_p 0.95 \
  --limit $LIMIT

echo "===== HF Temporal Counter Framing DONE ====="

