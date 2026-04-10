#!/bin/bash
#SBATCH --job-name  VividnessFraming-qwen
#SBATCH --time      1000:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate RoleConflict

MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')

BASE_DIR="/home/swiftie1230/EGO/FRAMING/FramingSensitivity"
INPUT_DIR="${BASE_DIR}/skeleton/data"
OUTPUT_DIR="${BASE_DIR}/framing/experiential_framing/output/v7"

mkdir -p "${OUTPUT_DIR}"

python src/VividnessFraming.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/vividness" \
  --model_tag "${MODEL_TAG}" \
  --backend hf \
  --hf_model_id "${MODEL}" \
  --temperature 0.1 \
  --max_tokens 700 \
  --domain_filter role_conflict,moral_dilemma,life_safety,decision_choice,legal_decision \
  --start_idx 0 \
  --limit 400