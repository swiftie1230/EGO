#!/bin/bash
#SBATCH --job-name  ValueMining-gpt_nano
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

TEMPERATURE=0.2
MAX_TOKENS=1500

BASE_DIR="/home/swiftie1230/EGO/FRAMING/FramingSensitivity"

INPUT_DIR="${BASE_DIR}/skeleton/data"

OUTPUT_DIR="${BASE_DIR}/framing/contextual_envelope_framing/value_mining/output/v6"
SCRIPT="${BASE_DIR}/framing/contextual_envelope_framing/value_mining/src/ValueMiningV2.py"

LOG_DIR="${BASE_DIR}/framing/contextual_envelope_framing/value_mining/logs"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/value_mining_or_${MODEL_TAG}_${TIMESTAMP}.log"

echo "======================================"
echo " Value Mining Experiment (OpenRouter)"
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
  --backend openrouter \
  --model "${MODEL}" \
  --model_tag "${MODEL_TAG}" \
  --temperature "${TEMPERATURE}" \
  --max_tokens "${MAX_TOKENS}" \
  --start_idx 0 \
  --limit 2 \
  --domain_filter role_conflict,moral_dilemma,life_safety,decision_choice,legal_decision \
  | tee "${LOG_FILE}"

echo "======================================"
echo " DONE"
echo "======================================"
