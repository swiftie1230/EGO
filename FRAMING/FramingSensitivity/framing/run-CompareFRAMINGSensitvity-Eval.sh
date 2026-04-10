#!/bin/bash
#SBATCH --job-name=FRAMING-sensitivity-C
#SBATCH --time=100:00:00
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH --gpus=0
#SBATCH --constraint=rtx02

source ~/.bashrc
conda activate RoleConflict

cd /home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing || exit 1

LIMIT=400

BASE_ROOT=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/v6
PARAPHRASE_ROOT=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/paraphrase/outputs/v3
RESULT_ROOT=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outputs

MODELS=(
  "Qwen_Qwen2.5-7B-Instruct"
  "mistralai_Mistral-7B-Instruct-v0.3"
  "meta-llama_Llama-3.1-8B-Instruct"
  #"google_gemma-2-9b-it"
)

FRAMINGS=(
  "temporal"
  #"experiential"
  #"value_tinted"
)

DATA_ITEMS=(
  "triage:triage"
  "roleconflict:roleconflict"
  "moralDilemma:ggb"
  "moralDilemma:unibench"
  "medical_triage:medical_triage"
  "SCOTUS:SCOTUS"
)

for MODEL in "${MODELS[@]}"; do
  for ITEM in "${DATA_ITEMS[@]}"; do
    DATASET="${ITEM%%:*}"
    STEM="${ITEM##*:}"
    MODEL_TAG=$(echo "$MODEL" | sed 's/[\/:]/_/g')

    BASE_JSONL="${BASE_ROOT}/${DATASET}/${STEM}_${MODEL_TAG}_preds_${LIMIT}.jsonl"

    if [ ! -f "$BASE_JSONL" ]; then
      echo "[SKIP] missing base: $BASE_JSONL"
      continue
    fi

    for FRAMING in "${FRAMINGS[@]}"; do
      case "$FRAMING" in
        temporal)
          ANCHOR_JSONL="${RESULT_ROOT}/outcome-oriented_framing/temporal_framing/${DATASET}/v1/${STEM}_${MODEL_TAG}_preds_${LIMIT}_temporal_counter.jsonl"
          PARAPHRASE_JSONL="${PARAPHRASE_ROOT}/${FRAMING}/${DATASET}/${STEM}_${MODEL_TAG}_preds_${LIMIT}_paraphrase_counter.jsonl"
          ;;
        experiential)
          ANCHOR_JSONL="${RESULT_ROOT}/experiential_framing/${DATASET}/v1/${STEM}_${MODEL_TAG}_preds_${LIMIT}_experiential_counter.jsonl"
          PARAPHRASE_JSONL="${PARAPHRASE_ROOT}/${FRAMING}/${DATASET}/${STEM}_${MODEL_TAG}_preds_${LIMIT}_paraphrase_counter.jsonl"
          ;;
        value_tinted)
          ANCHOR_JSONL="${RESULT_ROOT}/contextual_envelope_framing/value_tinted_narration/${DATASET}/v1/${STEM}_${MODEL_TAG}_preds_${LIMIT}_value_tinted_counter.jsonl"
          PARAPHRASE_JSONL="${PARAPHRASE_ROOT}/${FRAMING}/${DATASET}/${STEM}_${MODEL_TAG}_preds_${LIMIT}_paraphrase_counter.jsonl"
          ;;
        *)
          echo "[SKIP] unknown framing: $FRAMING"
          continue
          ;;
      esac

      if [ ! -f "$ANCHOR_JSONL" ]; then
        echo "[SKIP] missing anchor: $ANCHOR_JSONL"
        continue
      fi

      if [ ! -f "$PARAPHRASE_JSONL" ]; then
        echo "[SKIP] missing paraphrase: $PARAPHRASE_JSONL"
        continue
      fi

      echo "======================================================"
      echo "MODEL    : $MODEL"
      echo "DATASET  : $DATASET"
      echo "STEM     : $STEM"
      echo "FRAMING  : $FRAMING"
      echo "BASE     : $BASE_JSONL"
      echo "ANCHOR   : $ANCHOR_JSONL"
      echo "OTHER    : $PARAPHRASE_JSONL"
      echo "======================================================"

      python src/CompareFramingSensitivity.py \
        --base_jsonl "$BASE_JSONL" \
        --anchor_counter "$ANCHOR_JSONL" \
        --other_counters "$PARAPHRASE_JSONL" \
        --names paraphrase \
        --thresh 0.3

      echo
    done
  done
done