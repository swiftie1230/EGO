#!/bin/bash
#SBATCH --job-name  ValueTintedNarration-gpt_nano
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate RoleConflict


export OPENROUTER_API_KEY="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"

MODEL="openai/gpt-5.4-nano"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')
RUN_ID=2

SUFFIX="_${MODEL_TAG}"
if [ -n "$RUN_ID" ]; then
  SUFFIX="${SUFFIX}_limit${RUN_ID}"
fi

BASE_DIR="/home/swiftie1230/EGO/FRAMING/FramingSensitivity"
SCRIPT="${BASE_DIR}/framing/contextual_envelope_framing/value_tinted_narration/src/ValueTintedNarration.py"

# -----------------------------
# GGB
# -----------------------------
python "${SCRIPT}" \
  --skeleton_jsonl "${BASE_DIR}/skeleton/data/ggb_skeleton.jsonl" \
  --value_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_mining/output/v6/ggb_skeleton_values${SUFFIX}.jsonl" \
  --output_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_tinted_narration/output/v5/ggb_skeleton_value_tinted${SUFFIX}.jsonl" \
  --backend openrouter \
  --or_model "${MODEL}" \
  --temperature 0.2 \
  --max_tokens 650 \
  --limit $RUN_ID \
  --start_idx 0

# -----------------------------
# SUPER-SCOTUS (legal_decision)
# -----------------------------
python "${SCRIPT}" \
  --skeleton_jsonl "${BASE_DIR}/skeleton/data/SCOTUS_skeleton.jsonl" \
  --value_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_mining/output/v6/SCOTUS_skeleton_values${SUFFIX}.jsonl" \
  --output_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_tinted_narration/output/v5/SCOTUS_skeleton_value_tinted${SUFFIX}.jsonl" \
  --backend openrouter \
  --or_model "${MODEL}" \
  --temperature 0.2 \
  --max_tokens 650 \
  --limit $RUN_ID \
  --start_idx 0


# -----------------------------
# Medical Triage Alignment
# -----------------------------
python "${SCRIPT}" \
  --skeleton_jsonl "${BASE_DIR}/skeleton/data/medical_triage_alignment_skeleton.jsonl" \
  --value_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_mining/output/v6/medical_triage_alignment_skeleton_values${SUFFIX}.jsonl" \
  --output_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_tinted_narration/output/v5/medical_triage_alignment_skeleton_value_tinted${SUFFIX}.jsonl" \
  --backend openrouter \
  --or_model "${MODEL}" \
  --temperature 0.2 \
  --max_tokens 650 \
  --limit $RUN_ID \
  --start_idx 0


# -----------------------------
# UniBench
# -----------------------------
python "${SCRIPT}" \
  --skeleton_jsonl "${BASE_DIR}/skeleton/data/unibench_skeleton.jsonl" \
  --value_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_mining/output/v6/unibench_skeleton_values${SUFFIX}.jsonl" \
  --output_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_tinted_narration/output/v5/unibench_skeleton_value_tinted${SUFFIX}.jsonl" \
  --backend openrouter \
  --or_model "${MODEL}" \
  --temperature 0.2 \
  --max_tokens 650 \
  --limit $RUN_ID \
  --start_idx 0


# -----------------------------
# TRIAGE
# -----------------------------
#python "${SCRIPT}" \
#  --skeleton_jsonl "${BASE_DIR}/skeleton/data/triage_allocation.jsonl" \
#  --value_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_mining/output/v6/triage_allocation_values${SUFFIX}.jsonl" \
#  --output_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_tinted_narration/output/v5/triage_allocation_value_tinted${SUFFIX}.jsonl" \
#  --backend openrouter \
#  --model_id "${MODEL}" \
#  --temperature 0.2 \
#  --max_tokens 650 \
#  --limit $RUN_ID \
# --start_idx 0


# -----------------------------
# RoleConflict
# -----------------------------
python "${SCRIPT}" \
  --skeleton_jsonl "${BASE_DIR}/skeleton/data/roleconflict_allocation.jsonl" \
  --value_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_mining/output/v6/roleconflict_allocation_values${SUFFIX}.jsonl" \
  --output_jsonl "${BASE_DIR}/framing/contextual_envelope_framing/value_tinted_narration/output/v5/roleconflict_allocation_value_tinted${SUFFIX}.jsonl" \
  --backend openrouter \
  --or_model "${MODEL}" \
  --temperature 0.2 \
  --max_tokens 650 \
  --limit $RUN_ID \
  --start_idx 0

echo "======================================"
echo " Value-Tinted Narration (OpenRouter)"
echo " Model  : ${MODEL}"
echo " Run ID : ${RUN_ID}"
echo "======================================"

