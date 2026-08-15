# Final Report: Mistral 7B QLoRA Instruction Fine-Tuning

## Executive Summary

This project fine-tuned `mistralai/Mistral-7B-v0.3` on `yahma/alpaca-cleaned` using QLoRA on a single NVIDIA RTX 4060 Laptop GPU with 8GB VRAM. The final rank-8 adapter trained for 1,200 optimizer steps and reduced held-out validation perplexity from `4.97` to `2.62`.

The project is now portfolio-ready: it includes a reproducible training pipeline, a final saved adapter, base-vs-fine-tuned evaluation, qualitative generations, and real 7B ablations.

## Final Model

| Item | Value |
|---|---:|
| Base model | `mistralai/Mistral-7B-v0.3` |
| Dataset | `yahma/alpaca-cleaned` |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 + double quantization |
| Optimizer | `paged_adamw_8bit` |
| LoRA rank | `8` |
| LoRA alpha | `16` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Max sequence length | `1024` |
| Training steps | `1200` |
| Effective batch size | `16` |
| GPU | RTX 4060 Laptop GPU, 8GB VRAM |
| Observed peak VRAM | about `7.5GB` |
| Trainable parameters | `6,815,744` |
| Trainable fraction | `0.0939%` |
| Frozen parameter reduction | `99.9061%` vs updating all 7.25B params |

Final adapter:

```text
outputs/mistral7b-alpaca-cleaned-qlora-r8
```

## Quantitative Evaluation

Evaluation was run on the same deterministic 1,000-example held-out validation split.

| Model | Eval Loss | Perplexity | Samples |
|---|---:|---:|---:|
| Base Mistral 7B | `1.6029` | `4.9675` | `1000` |
| QLoRA adapter | `0.9628` | `2.6190` | `1000` |

Improvement:

| Metric | Improvement |
|---|---:|
| Eval loss reduction | `39.93%` |
| Perplexity reduction | `47.28%` |
| Base / adapter perplexity ratio | `1.90x` |

Training-time validation loss improved steadily:

| Step | Eval Loss |
|---:|---:|
| 100 | `1.0143` |
| 200 | `0.9968` |
| 300 | `0.9886` |
| 400 | `0.9833` |
| 500 | `0.9780` |
| 600 | `0.9741` |
| 700 | `0.9697` |
| 800 | `0.9672` |
| 900 | `0.9648` |
| 1000 | `0.9634` |
| 1100 | `0.9629` |
| 1200 | `0.9628` |

Interpretation: validation improved monotonically but flattened after step 900, so 900-1200 steps is the useful stopping region for this dataset/config on the available GPU.

## Ablations

Short 100-step ablations were run on the same model/dataset/split to compare early learning behavior.

| Experiment | Target Modules | Rank | Trainable Params | Eval Loss @ 100 Steps |
|---|---|---:|---:|---:|
| Rank 4 | q/k/v/o | 4 | `3.41M` | `1.0312` |
| Rank 8 final run | q/k/v/o | 8 | `6.82M` | `1.0143` |
| Rank 16 | q/k/v/o | 16 | `13.63M` | `1.0106` |
| q/v only | q/v | 8 | `3.41M` | `1.0353` |

Findings:

- Rank 16 learned fastest at 100 steps but used 4x the trainable parameters of rank 4 and 2x rank 8.
- Rank 8 was the best practical default: much stronger than rank 4/qv-only, close to rank 16 early, and cheaper to continue for the full 1,200-step run.
- q/v-only underperformed q/k/v/o at the same 3.41M parameter budget, suggesting that adapting all attention projections was worthwhile for this instruction-tuning setup.

## Qualitative Evaluation

Fixed prompts were saved in:

```text
reports/sample_prompts.json
```

Generated outputs were saved in:

```text
reports/generation_base.json
reports/generation_adapter.json
```

Observed pattern:

- The fine-tuned model generally follows the Alpaca response format more directly.
- The adapter improves concise diagnostic prompts, especially the overfitting/perplexity prompt.
- The model still hallucinates some technical specifics in LoRA parameter-count prompts. This is a useful limitation: instruction fine-tuning on Alpaca improved style and validation likelihood, but did not make the model a reliable LoRA theory calculator.

Honest conclusion: this is a strong PEFT systems project, not a claim that the model became a domain expert.

## Reproducibility

Train final adapter:

```bash
python -m src.train --config configs/t4_mistral_alpaca_qlora.json
```

Resume from checkpoint:

```bash
python -m src.train \
  --config configs/t4_mistral_alpaca_qlora.json \
  --resume-from-checkpoint outputs/mistral7b-alpaca-cleaned-qlora-r8/checkpoint-600
```

Evaluate base:

```bash
python -m src.evaluate \
  --config configs/t4_mistral_alpaca_qlora.json \
  --max-eval-samples 1000 \
  --output reports/eval_base_1000.json
```

Evaluate adapter:

```bash
python -m src.evaluate \
  --config configs/t4_mistral_alpaca_qlora.json \
  --adapter-dir outputs/mistral7b-alpaca-cleaned-qlora-r8 \
  --max-eval-samples 1000 \
  --output reports/eval_adapter_1000.json
```

Run ablations:

```bash
python -m src.train --config configs/ablate_rank4_100steps.json
python -m src.train --config configs/ablate_rank16_100steps.json
python -m src.train --config configs/ablate_qv_100steps.json
```

Generate samples:

```bash
python -m src.batch_generate \
  --config configs/t4_mistral_alpaca_qlora.json \
  --adapter-dir outputs/mistral7b-alpaca-cleaned-qlora-r8 \
  --prompts reports/sample_prompts.json \
  --output reports/generation_adapter.json
```

## Resume Bullets

- Fine-tuned `Mistral-7B-v0.3` with QLoRA rank-8 on Alpaca-cleaned, training only `6.82M` parameters (`0.0939%` of 7.25B weights) while freezing `99.9061%` of the model.
- Implemented 4-bit NF4 QLoRA with double quantization and paged AdamW, fitting Mistral 7B fine-tuning on a single RTX 4060 Laptop GPU under about `7.5GB` observed VRAM.
- Reduced held-out validation perplexity from `4.97` to `2.62` across 1,000 validation examples, a `47.28%` perplexity reduction over the base model.
- Ablated LoRA rank and target modules on real 7B runs; rank-16 reached `1.0106` eval loss at 100 steps vs rank-8 `1.0143`, while q/v-only underperformed at `1.0353`, validating q/k/v/o rank-8 as the practical full-run choice.

## Interview Talking Points

- LoRA trains a low-rank update `Delta W = (alpha / r) BA` instead of the full matrix.
- QLoRA saves memory by storing frozen base weights in 4-bit NF4 while backpropagating through trainable adapters.
- The project used q/k/v/o rather than q/v-only because the ablation showed better early validation loss.
- Rank 16 improved early validation loss slightly but doubled trainable parameters relative to rank 8.
- Perplexity improved strongly, but qualitative outputs still show hallucinated technical counts; evaluation should not overclaim.

## Limitations

- No MMLU/AlpacaEval score was run locally because Windows long-path issues blocked `lm-eval` installation in the original environment.
- Alpaca-cleaned is useful for instruction formatting but not ideal for factual domain expertise.
- The ablations are short 100-step comparisons, not fully converged sweeps.
- Qualitative outputs show that instruction tuning improved response style more than technical factuality.

## Verdict

This is now a resume-ready LoRA/QLoRA fine-tuning project. The strongest claims are about systems execution, memory-efficient fine-tuning, reproducible evaluation, and measured ablations.
