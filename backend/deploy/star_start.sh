#!/usr/bin/env bash
# Launch the backend on the star box (Linux + GPU). Run inside tmux:
#   tmux new-session -d -s cs '~/code-summary-artifact/backend/deploy/star_start.sh 2>&1 | tee ~/cs_backend.log'
set -e
cd "$(dirname "$0")/.."   # -> backend/

export CS_CORPUS_PATH="${CS_CORPUS_PATH:-$HOME/code_sum_rag/external/codexglue_python_train.jsonl}"
export CS_CORPUS_LIMIT="${CS_CORPUS_LIMIT:-30000}"

# SelectorNet core-block classifier (transferred from myci) + CodeBERT encoder.
export CS_EXTRACTOR_WEIGHTS="${CS_EXTRACTOR_WEIGHTS:-$HOME/code-summary-artifact/backend/assets/extractor/pytorch_model.bin}"
export CS_CODEBERT_PATH="${CS_CODEBERT_PATH:-$HOME/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39}"
export CS_DEVICE="${CS_DEVICE:-cuda}"

exec "${CS_PYTHON:-$HOME/anaconda3/envs/L2NL/bin/python}" -m uvicorn app.main:app \
    --host 0.0.0.0 --port 8000
