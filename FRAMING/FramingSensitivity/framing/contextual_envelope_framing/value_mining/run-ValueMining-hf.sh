#!/bin/bash
#SBATCH --job-name  ValueMining-qwen
#SBATCH --time      1000:00:00
#SBATCH -c          4
#SBATCH --mem       70G
#SBATCH --gpus      1
#SBATCH --constraint=rtx

source ~/.bashrc
conda activate RoleConflict

MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')

TEMPERATURE=0
MAX_TOKENS=1500

BASE_DIR="/home/swiftie1230/EGO/FRAMING/FramingSensitivity"

INPUT_DIR="${BASE_DIR}/skeleton/data"

OUTPUT_DIR="${BASE_DIR}/framing/contextual_envelope_framing/value_mining/output/v5"
SCRIPT="${BASE_DIR}/framing/contextual_envelope_framing/value_mining/src/ValueMiningV2.py"

LOG_DIR="${BASE_DIR}/framing/contextual_envelope_framing/value_mining/logs"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/value_mining_${MODEL_TAG}_${TIMESTAMP}.log"

echo "======================================"
echo " Value Mining Experiment (Schwartz)"
echo "--------------------------------------"
echo " Model        : ${MODEL}"
echo " Temperature  : ${TEMPERATURE}"
echo " Max Tokens   : ${MAX_TOKENS}"
echo " Input Dir    : ${INPUT_DIR}"
echo " Output Dir   : ${OUTPUT_DIR}"
echo " Log File     : ${LOG_FILE}"
echo "======================================"

python "${SCRIPT}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --backend hf \
  --model_tag "${MODEL_TAG}" \
  --hf_model_id "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --hf_max_new_tokens "${MAX_TOKENS}" \
  --domain_filter role_conflict,moral_dilemma,life_safety,decision_choice,legal_decision \
  --start_idx 0 \
  --limit 400 \
  | tee "${LOG_FILE}"

echo "======================================"
echo " DONE"
echo "======================================"
