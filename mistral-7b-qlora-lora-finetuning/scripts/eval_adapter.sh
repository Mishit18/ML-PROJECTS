#!/usr/bin/env bash
set -euo pipefail

ADAPTER_DIR="${1:-outputs/mistral7b-alpaca-cleaned-qlora-r8}"
python -m src.evaluate \
  --config configs/t4_mistral_alpaca_qlora.json \
  --adapter-dir "$ADAPTER_DIR" \
  --max-eval-samples 500 \
  --output reports/eval_adapter.json
