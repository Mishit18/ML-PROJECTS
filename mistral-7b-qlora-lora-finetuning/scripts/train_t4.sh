#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/t4_mistral_alpaca_qlora.json
