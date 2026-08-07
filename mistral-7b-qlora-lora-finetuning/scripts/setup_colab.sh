#!/usr/bin/env bash
set -euo pipefail

pip install -U pip
pip install -r requirements.txt
accelerate config default
