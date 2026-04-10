#!/bin/bash
#SBATCH --job-name  VividnessFraming-gpt_nano
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate RoleConflict

MODEL="openai/gpt-5.4-nano"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')

export OPENROUTER_API_KEY="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"

BASE_DIR="/home/swiftie1230/EGO/FRAMING/FramingSensitivity"
INPUT_DIR="${BASE_DIR}/skeleton/data"
OUTPUT_DIR="${BASE_DIR}/framing/experiential_framing/output/v8"

mkdir -p "${OUTPUT_DIR}"

python src/VividnessFraming.py \
  --backend openrouter \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/vividness" \
  --model_tag "${MODEL_TAG}" \
  --model "${MODEL}" \
  --temperature 0.1 \
  --max_tokens 700 \
  --domain_filter role_conflict,moral_dilemma,life_safety,decision_choice,legal_decision \
  --start_idx 0 \
  --limit 2