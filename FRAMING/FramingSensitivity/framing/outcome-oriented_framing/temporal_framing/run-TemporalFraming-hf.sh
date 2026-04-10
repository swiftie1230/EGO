#!/bin/bash

#SBATCH --job-name  TemporalFraming-qwen
#SBATCH --time      1000:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=ada02

source ~/.bashrc
conda activate RoleConflict

export CUDA_VISIBLE_DEVICES=1

# 모델 설정
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')
RUN_ID=400

SUFFIX="_${MODEL_TAG}"
if [ -n "$RUN_ID" ]; then
  SUFFIX="${SUFFIX}_${RUN_ID}"
fi


# Unibench
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/unibench_skeleton.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v4/unibench_skeleton_temporal_framing${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID

# Traige
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/triage_allocation.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v4/triage_allocation_temporal_framing${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID

# GGB
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v4/ggb_skeleton_temporal_framing${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID
  
# RoleConflict
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/roleconflict_allocation.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v4/roleconflict_allocation_temporal_framing${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --max_tokens 650 \
  --start_idx 0 \
  --limit $RUN_ID
  
# SCOTUS
python /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/src/TemporalFraming.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/SCOTUS_skeleton.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outcome-oriented_framing/temporal_framing/output/v4/SCOTUS_skeleton_temporal_framing${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct \
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