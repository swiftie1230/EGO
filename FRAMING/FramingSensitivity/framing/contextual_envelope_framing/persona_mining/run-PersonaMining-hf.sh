#!/bin/bash
#SBATCH --job-name  PersonaMining-qwen
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=ada02

source ~/.bashrc
conda activate RoleConflict


MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')
TEMPERATURE=0
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
  --backend hf \
  --model_tag "${MODEL_TAG}" \
  --hf_model_id "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --hf_max_new_tokens "${MAX_TOKENS}" \
  --domain_filter life_safety,moral_dilemma,decision_choice \
  | tee "${LOG_FILE}"

echo "======================================"
echo " DONE"
echo "======================================"