#!/bin/bash

#SBATCH --job-name  TemporalFraming-gpt_nano
#SBATCH --time      1000:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=ada02

source ~/.bashrc
conda activate RoleConflict

export CUDA_VISIBLE_DEVICES=1

export OPENROUTER_API_KEY="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"

# 모델 설정
MODEL="openai/gpt-5.4-nano"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')
RUN_ID=2

SUFFIX="_${MODEL_TAG}"
if [ -n "$RUN_ID" ]; then
  SUFFIX="${SUFFIX}_${RUN_ID}"
fi


# Unibench
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/unibench_skeleton.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v5/unibench_skeleton_temporal_framing${SUFFIX}.jsonl \
  --backend openrouter \
  --or_model "${MODEL}" \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID

# Traige
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/triage_allocation.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v5/triage_allocation_temporal_framing${SUFFIX}.jsonl \
  --backend openrouter \
  --or_model "${MODEL}" \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID

# GGB
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v5/ggb_skeleton_temporal_framing${SUFFIX}.jsonl \
  --backend openrouter \
  --or_model "${MODEL}" \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID
  
# RoleConflict
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/roleconflict_allocation.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v5/roleconflict_allocation_temporal_framing${SUFFIX}.jsonl \
  --backend openrouter \
  --or_model "${MODEL}" \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID
  
# SCOTUS
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/SCOTUS_skeleton.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v5/SCOTUS_skeleton_temporal_framing${SUFFIX}.jsonl \
  --backend openrouter \
  --or_model "${MODEL}" \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID
  
# MedicalTriageAlignment
#python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
#  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/medical_triage_alignment_skeleton.jsonl \
#  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v1/medical_triage_alignment_skeleton_temporal_framing${SUFFIX}.jsonl \
#  --backend hf \
#  --model_id Qwen/Qwen2.5-7B-Instruct \
#  --max_tokens 650 \
#  --start_idx 0 \
#  --limit $RUN_ID