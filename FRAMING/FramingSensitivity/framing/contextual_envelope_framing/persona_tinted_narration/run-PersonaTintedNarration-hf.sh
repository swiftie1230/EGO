#!/bin/bash
#SBATCH --job-name  PersonaTintedNarration-qwen
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate RoleConflict

MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')
RUN_ID=5

SUFFIX="_${MODEL_TAG}"
if [ -n "$RUN_ID" ]; then
  SUFFIX="${SUFFIX}_${RUN_ID}"
fi

python src/PersonaTintedNarration.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton.jsonl \
  --persona_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/ggb_skeleton_personas_Qwen_Qwen2.5-7B-Instruct_5.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_tinted_narration/output/v1/ggb_skeleton_persona_tinted${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --temperature 0.2 \
  --max_tokens 650

python src/PersonaTintedNarration.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/triage_allocation.jsonl \
  --persona_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/triage_allocation_personas_Qwen_Qwen2.5-7B-Instruct_5.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_tinted_narration/output/v1/triage_allocation_persona_tinted${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct

python src/PersonaTintedNarration.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/unibench_skeleton.jsonl \
  --persona_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/unibench_skeleton_personas_Qwen_Qwen2.5-7B-Instruct_5.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_tinted_narration/output/v1/unibench_skeleton_persona_tinted${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct