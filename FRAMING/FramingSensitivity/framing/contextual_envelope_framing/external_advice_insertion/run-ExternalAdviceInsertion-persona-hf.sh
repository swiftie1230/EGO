#!/bin/bash
#SBATCH --job-name  ExternalAdviceInsertion-qwen
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=ada03

source ~/.bashrc
conda activate RoleConflict

MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')
# RUN_ID=5

SUFFIX="_${MODEL_TAG}"
if [ -n "$RUN_ID" ]; then
  SUFFIX="${SUFFIX}_${RUN_ID}"
fi

python src/ExternalAdviceInsertion-persona.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton.jsonl \
  --persona_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/ggb_skeleton_personas_Qwen_Qwen2.5-7B-Instruct.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/external_advice_insertion/output/v3/ggb_skeleton_external_advice_inserted${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --max_tokens 650

python src/ExternalAdviceInsertion-persona.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/triage_allocation.jsonl \
  --persona_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/triage_allocation_personas_Qwen_Qwen2.5-7B-Instruct.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/external_advice_insertion/output/v3/triage_allocation_external_advice_inserted${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct
  --max_tokens 650

python src/ExternalAdviceInsertion-persona.py \
  --skeleton_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/unibench_skeleton.jsonl \
  --persona_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/unibench_skeleton_personas_Qwen_Qwen2.5-7B-Instruct.jsonl \
  --output_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/external_advice_insertion/output/v3/unibench_skeleton_external_advice_inserted${SUFFIX}.jsonl \
  --backend hf \
  --model_id Qwen/Qwen2.5-7B-Instruct
  --max_tokens 650