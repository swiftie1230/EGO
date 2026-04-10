#!/bin/bash
#SBATCH --job-name  FRAMING-sensitivity
#SBATCH --time      100:00:00
#SBATCH -c          4
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint=rtx

source ~/.bashrc
conda activate RoleConflict

PRED_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/v3
FRAMING_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/contextual_envelope_framing/value_tinted_narration/output/v1
OUTDIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outputs/contextual_envelope_framing/value_tinted_narration
PARAPHRASE_DIR=/home/swiftie1230/EGO/FRAMING/FramingSensitivity/paraphrase/data

#--counter_jsonl $OUT_DIR/moralDilemma/v4/unibench_Qwen_Qwen2.5-7B-Instruct_preds_200.jsonl
#  --counter_jsonl $OUTDIR/moralDilemma/v4/ggb_Qwen_Qwen2.5-7B-Instruct_preds_200_value_tinted_counter.jsonl \

python src/FramingSensitivity2.py \
  --base_jsonl /home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/v2/moralDilemma/ggb_Qwen_Qwen2.5-7B-Instruct_preds_200.jsonl \
  --counter_jsonl $PARAPHRASE_DIR/ggb_skeleton.expand.jsonl \
  --distance l1 \
  --thresh 0.3 \
  --group_by axis value_type
