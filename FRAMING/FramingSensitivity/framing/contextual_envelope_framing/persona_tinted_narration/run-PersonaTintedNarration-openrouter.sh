#!/bin/bash
#SBATCH --job-name  PersonaTintedNarration
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate RoleConflict


export OPENROUTER_API_KEY="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"

MODEL="openai/gpt-4.1-mini"
MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')
RUN_ID=5

SUFFIX="_${MODEL_TAG}"
if [ -n "$RUN_ID" ]; then
  SUFFIX="${SUFFIX}_${RUN_ID}"
fi

python PersonaTintedNarration.py \
  --skeleton_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton.jsonl \
  --persona_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/ggb_skeleton_personas_Qwen_Qwen2.5-7B-Instruct_5.jsonl \
  --output_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_tinted_narration/output/v1/ggb_skeleton_persona_tinted${SUFFIX}.jsonl \
  --backend openrouter \
  --or_api_key "${OPENROUTER_API_KEY}" \
  --or_model openai/gpt-4.1-mini \
  --temperature 0.2 \
  --max_tokens 650

python PersonaTintedNarration.py \
  --skeleton_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/triage_allocation.jsonl \
  --persona_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/triage_allocation_personas_Qwen_Qwen2.5-7B-Instruct_5.jsonl \
  --output_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_tinted_narration/output/v1/triage_allocation_persona_tinted${SUFFIX}.jsonl \
  --backend openrouter \
  --or_api_key "${OPENROUTER_API_KEY}" \
  --or_model openai/gpt-4.1-mini \
  --temperature 0.15 \
  --max_tokens 550

python PersonaTintedNarration.py \
  --skeleton_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/unibench_skeleton.jsonl \
  --persona_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_mining/output/v1/unibench_skeleton_personas_Qwen_Qwen2.5-7B-Instruct_5.jsonl \
  --output_jsonl \
    /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/persona_tinted_narration/output/v1/unibench_skeleton_persona_tinted${SUFFIX}.jsonl \
  --backend openrouter \
  --or_api_key "${OPENROUTER_API_KEY}" \
  --or_model openai/gpt-4.1-mini \
  --temperature 0.25 \
  --max_tokens 600
