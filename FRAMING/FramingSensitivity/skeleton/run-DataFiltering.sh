#!/bin/bash
#SBATCH --job-name  SkeletonFiltering
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       45G
#SBATCH --gpus      1
#SBATCH --constraint=rtx

conda activate RoleConflict

#python src/DataFiltering.py \
#  --dataset GGB \
#  --file_a /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton_gpt-4.1-mini.jsonl \
#  --file_b /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton_qwen2.5-72b-inst.jsonl \
#  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton.jsonl \
#  --audit_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton_audit.jsonl \
#  --model_name Qwen/Qwen2.5-7B-Instruct \
#  --device cuda \
#  --torch_dtype bfloat16 \
#  --max_new_tokens 32
  
python src/DataFiltering.py \
  --dataset SUPER_SCOTUS \
  --file_a /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/SCOTUS_skeleton_gpt-4.1-mini.jsonl \
  --file_b /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/SCOTUS_skeleton_qwen2.5-72b-inst.jsonl \
  --output_jsonl SCOTUS_skeleton.jsonl \
  --audit_jsonl scotus_selected_qwen7b_judge.audit.jsonl \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --torch_dtype bfloat16 \
  --max_new_tokens 32