#!/bin/bash
#SBATCH --job-name  Paraphrase-generation-qwen
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=ada

source ~/.bashrc
conda activate RoleConflict

# -----------------------------
# SETTINGS
# -----------------------------
BACKEND=hf

#MODEL=meta-llama/Llama-3.1-8B-Instruct
#MODEL=mistralai/Mistral-7B-Instruct-v0.3 
MODEL=Qwen/Qwen2.5-7B-Instruct 
#MODEL=google/gemma-2-9b-it

MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')

# Base predictions (same as VT)
PRED_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/v6

# Original skeleton (id = ggb_GGB_1 etc.) for prompt building
# (VT에서는 framing_path로 skeleton+framings를 읽었는데, paraphrase baseline은 원본 skeleton을 읽음)
BASE_SKELETON_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data

# Paraphrase expanded rows (meta.paraphrase_of / meta.paraphrase_field)
PARAPHRASE_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/paraphrase/data
PARAPHRASE_PATTERN="*.expand.jsonl"

# Output value_tinted,experiential,temporal
AXIS=temporal
OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/paraphrase/outputs/v3


# Decoding (match VT)
NBEST=10
DECODE_MODE=sample
TEMP=0.7
TOP_P=0.95
BATCH=4

# Optional
# LIMIT=5

# ==========================================
# TRIAGE
# ==========================================
python src/ParaphraseGeneration.py \
  --pred_dir $PRED_DIR/triage \
  --axis $AXIS \
  --model_tag $MODEL_TAG \
  --dataset_prefix triage \
  --base_skeleton_path $BASE_SKELETON_DIR/triage_allocation.jsonl \
  --paraphrase_path $PARAPHRASE_DIR \
  --paraphrase_pattern "$PARAPHRASE_PATTERN" \
  --out_path $OUTDIR/$AXIS/triage \
  --model_id $MODEL \
  --batch_size $BATCH \
  --decode_mode $DECODE_MODE \
  --nbest $NBEST \
  --temperature $TEMP \
  --top_p $TOP_P
#  --limit $LIMIT

# ==========================================
# ROLECONFLICT
# ==========================================
python src/ParaphraseGeneration.py \
  --pred_dir $PRED_DIR/roleconflict \
  --axis $AXIS \
  --model_tag $MODEL_TAG \
  --dataset_prefix roleconflict \
  --base_skeleton_path $BASE_SKELETON_DIR/roleconflict_allocation.jsonl \
  --paraphrase_path $PARAPHRASE_DIR \
  --paraphrase_pattern "$PARAPHRASE_PATTERN" \
  --out_path $OUTDIR/$AXIS/roleconflict \
  --model_id $MODEL \
  --batch_size $BATCH \
  --decode_mode $DECODE_MODE \
  --nbest $NBEST \
 --temperature $TEMP \
  --top_p $TOP_P
##  --limit $LIMIT


# ==========================================
# medical_triage
# ==========================================
#python src/ParaphraseGeneration.py \
#  --pred_dir $PRED_DIR/medical_triage \
#  --axis $AXIS \
#  --model_tag $MODEL_TAG \
#  --dataset_prefix medical_triage \
#  --base_skeleton_path $BASE_SKELETON_DIR/medical_triage_alignment_skeleton.jsonl \
#  --paraphrase_path $PARAPHRASE_DIR \
#  --paraphrase_pattern "$PARAPHRASE_PATTERN" \
#  --out_path $OUTDIR/$AXIS/medical_triage \
#  --model_id $MODEL \
#  --batch_size $BATCH \
#  --decode_mode $DECODE_MODE \
#  --nbest $NBEST \
# --temperature $TEMP \
#  --top_p $TOP_P
##  --limit $LIMIT


# ==========================================
# SCOTUS
# ==========================================
python src/ParaphraseGeneration.py \
  --pred_dir $PRED_DIR/SCOTUS \
  --axis $AXIS \
  --model_tag $MODEL_TAG \
  --dataset_prefix SCOTUS \
  --base_skeleton_path $BASE_SKELETON_DIR/SCOTUS_skeleton.jsonl \
  --paraphrase_path $PARAPHRASE_DIR \
  --paraphrase_pattern "$PARAPHRASE_PATTERN" \
  --out_path $OUTDIR/$AXIS/SCOTUS \
  --model_id $MODEL \
  --batch_size $BATCH \
  --decode_mode $DECODE_MODE \
  --nbest $NBEST \
 --temperature $TEMP \
  --top_p $TOP_P
##  --limit $LIMIT

# ==========================================
# GGB (moralDilemma)
# ==========================================
python src/ParaphraseGeneration.py \
  --pred_dir /$PRED_DIR/moralDilemma \
  --axis $AXIS \
  --model_tag $MODEL_TAG \
  --dataset_prefix ggb \
  --base_skeleton_path $BASE_SKELETON_DIR/ggb_skeleton.jsonl \
  --paraphrase_path $PARAPHRASE_DIR \
  --paraphrase_pattern "$PARAPHRASE_PATTERN" \
  --out_path $OUTDIR/$AXIS/moralDilemma \
  --model_id $MODEL \
  --batch_size $BATCH \
  --decode_mode $DECODE_MODE \
  --nbest $NBEST \
  --temperature $TEMP \
  --top_p $TOP_P
#  --limit $LIMIT

# ==========================================
# UNIBENCH
# ==========================================
python src/ParaphraseGeneration.py \
  --pred_dir $PRED_DIR/moralDilemma \
  --axis $AXIS \
  --model_tag $MODEL_TAG \
  --dataset_prefix unibench \
  --base_skeleton_path $BASE_SKELETON_DIR/unibench_skeleton.jsonl \
  --paraphrase_path $PARAPHRASE_DIR \
  --paraphrase_pattern "$PARAPHRASE_PATTERN" \
  --out_path $OUTDIR/$AXIS/moralDilemma \
  --model_id $MODEL \
  --batch_size $BATCH \
  --decode_mode $DECODE_MODE \
  --nbest $NBEST \
  --temperature $TEMP \
  --top_p $TOP_P
#  --limit $LIMIT

echo "===== HF Paraphrase Baseline DONE ====="