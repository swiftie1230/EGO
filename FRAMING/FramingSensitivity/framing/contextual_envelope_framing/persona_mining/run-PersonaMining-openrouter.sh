#!/bin/bash
#SBATCH --job-name  PersonaMining
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate RoleConflict


export OPENROUTER_API_KEY="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"

MODEL="openai/gpt-4.1-mini"
TEMPERATURE=0.2
MAX_TOKENS=1500

BASE_DIR="/home/swiftie1230/EGO/FRAMING/FramingSensitivity"

INPUT_DIR="${BASE_DIR}/skeleton/data"
OUTPUT_DIR="${BASE_DIR}/framing/contextual_envelope_framing/persona_mining/output/v1"

SCRIPT="${BASE_DIR}/framing/contextual_envelope_framing/persona_mining/src/PersonaMining.py"

LOG_DIR="${BASE_DIR}/framing/contextual_envelope_framing/persona_mining/logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/persona_mining_${TIMESTAMP}.log"


echo "======================================"
echo " Persona Mining Experiment"
echo "--------------------------------------"
echo " Model        : ${MODEL}"
echo " Temperature  : ${TEMPERATURE}"
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
  --domain_filter life_safety,moral_dilemma \
  | tee "${LOG_FILE}"

echo "======================================"
echo " DONE"
echo "======================================"