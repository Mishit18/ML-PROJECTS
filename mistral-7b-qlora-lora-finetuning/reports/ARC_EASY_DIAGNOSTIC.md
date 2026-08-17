# ARC-Easy Base vs Adapter Diagnostic

## Protocol

- Harness: `lm-eval` 0.4.8
- Task: ARC-Easy, zero-shot
- Sample: first 100 evaluation items selected by the harness under seed 42
- Inference: Mistral-7B-v0.3 loaded in 4-bit mode, batch size 1
- Comparison: frozen base model vs the rank-8 Alpaca-cleaned QLoRA adapter

## Results

| Model | Raw accuracy | Length-normalized accuracy |
|---|---:|---:|
| Base Mistral-7B-v0.3 | 78% | 76% |
| QLoRA adapter | 75% | 79% |

The adapter improved normalized accuracy by 3 percentage points while raw accuracy fell by 3 points. With only 100 items and standard errors near 4 percentage points, this is a diagnostic rather than evidence of a statistically reliable reasoning gain. The defensible conclusion is that Alpaca instruction tuning substantially improved held-out instruction likelihood and did not show a clear general-reasoning improvement on this small ARC-Easy check.

Raw `lm-eval` outputs are committed under `reports/lm_eval_base_arc_easy.json/` and `reports/lm_eval_adapter_arc_easy.json/`.
