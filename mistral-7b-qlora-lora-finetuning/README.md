# Mistral 7B LoRA / QLoRA Fine-Tuning Project

This project is a resume-grade, reproducible LoRA and QLoRA fine-tuning setup for a single GPU. It is designed for a first credible run on Colab T4 and stronger ablations on A100.

## Final Results

The completed run fine-tuned `mistralai/Mistral-7B-v0.3` with QLoRA rank 8 on `yahma/alpaca-cleaned` for 1,200 steps on a single RTX 4060 Laptop GPU with 8GB VRAM.

| Metric | Result |
|---|---:|
| Trainable parameters | `6,815,744` |
| Trainable fraction | `0.0939%` |
| Base validation perplexity | `4.9675` |
| Fine-tuned validation perplexity | `2.6190` |
| Perplexity reduction | `47.28%` |
| Final eval loss | `0.9628` |

See the full write-up in `reports/FINAL_REPORT.md`.

## Tech Resume Screening Summary

Verified locally:
- Pytest suite passes 2/2 tests for the manual LoRA implementation.
- Final 1,200-step QLoRA run trained 6,815,744 adapter parameters, only 0.0939% of the 7.25B-class base model.
- Held-out validation perplexity improved from 4.9675 to 2.6190 across 1,000 validation samples, a 47.28% reduction.
- Real 7B ablations compare rank-4, rank-8, rank-16, and q/v-only target modules at 100 steps.
- Project is strongest as a PEFT systems and evaluation project, not as a claim of production-safe LLM deployment.

## Recommendation

Use `mistralai/Mistral-7B-v0.3` with QLoRA on `yahma/alpaca-cleaned` for the first run.

Why this is the best first project choice:

- Mistral 7B has stronger resume signal than Phi-2 because it proves you can fine-tune a real 7B-class model under memory constraints.
- Mistral is easier to use than LLaMA-2 for many students because it does not require Meta gated access approval.
- Alpaca-cleaned is small enough for a sub-4-hour T4 run and has a simple instruction/input/output schema, which makes debugging easier.
- OpenHermes 2.5 is a better second-stage dataset on A100 because it is larger and higher quality, but it is too large for a clean first T4 run unless you subsample.
- Phi-2 is a good fallback when GPU access is weak, but it is less impressive if your goal is parameter-efficient tuning depth on 7B models.

Recommended progression:

1. T4 baseline: `Mistral-7B-v0.3` + `alpaca-cleaned`, QLoRA rank 8, 1,200 steps.
2. A100 upgrade: `Mistral-7B-v0.3` + 100k to 300k OpenHermes samples, QLoRA rank 16.
3. Resume ablations: ranks 4/8/16/32, target modules, alpha scaling, QLoRA vs unquantized LoRA where hardware permits.

## LoRA Theory

For a frozen dense projection `W` with shape `(d, k)`, LoRA trains a low-rank update:

```text
W' = W + (alpha / r) * B A
```

where:

- `A` has shape `(r, k)`
- `B` has shape `(d, r)`
- `r << min(d, k)`
- `alpha / r` controls update scale

A full 4096 by 4096 projection has 16,777,216 parameters. A rank-8 LoRA update has:

```text
r * (d + k) = 8 * (4096 + 4096) = 65,536
```

That is about 0.39% of that layer, a 99.61% reduction for the adapted matrix. Across a 7B model, the trainable fraction is usually around 0.05% to 0.5% depending on target modules, which is why a resume bullet should report your measured trainable parameter count from `model.print_trainable_parameters()`.

### Target Modules

Use this order:

- `q_proj`, `v_proj`: smallest and classic LoRA setup, but may underfit instruction tuning.
- `q_proj`, `k_proj`, `v_proj`, `o_proj`: best first T4 choice.
- attention plus `gate_proj`, `up_proj`, `down_proj`: stronger but more trainable parameters and memory.

### Rank and Alpha

Rank controls adapter capacity. Alpha controls update magnitude through `alpha / r`.

Good defaults:

- `r=8`, `alpha=16` for T4.
- `r=16`, `alpha=32` for A100 or higher-quality dataset runs.
- Keep `alpha/r = 2` as the first stable setting, then ablate `1`, `2`, and `4`.

The manual implementation is in `src/manual_lora.py`; production training uses PEFT in `src/model.py`.

## QLoRA Theory

QLoRA stores the frozen base model in 4-bit precision while training small LoRA adapters. NF4 is used because pretrained neural weights are often approximately normally distributed, and NF4 allocates quantization levels to fit that distribution better than uniform int4.

Double quantization quantizes the quantization constants themselves, shaving additional memory. Paged optimizers use unified memory behavior to reduce temporary optimizer-state spikes, which is useful on T4 where a single spike can trigger out-of-memory.

Expected memory:

| Setup | Approx Peak VRAM |
|---|---:|
| 7B full fine-tune fp16 | 80GB+ |
| 7B LoRA fp16 base | 28GB to 40GB |
| 7B QLoRA NF4 rank 8, seq 1024 | 11GB to 15GB |
| 7B QLoRA NF4 rank 16, seq 2048 | 18GB to 28GB |

Actual VRAM depends on sequence length, batch size, target modules, checkpointing, and CUDA allocator behavior. Log your measured peak with `nvidia-smi`.

## Project Structure

```text
.
|-- configs/
|   |-- t4_mistral_alpaca_qlora.json
|   |-- a100_mistral_openhermes_qlora.json
|   `-- ablation_grid.json
|-- src/
|   |-- config.py
|   |-- data.py
|   |-- manual_lora.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   |-- infer.py
|   `-- ablate.py
|-- reports/
|   |-- experiment_log_template.md
|   `-- resume_bullets_template.md
|-- scripts/
|-- tests/
`-- requirements.txt
```

## Setup

On Colab or Linux GPU:

```bash
git clone <your-repo-url>
cd mistral-7b-qlora-lora-finetuning
bash scripts/setup_colab.sh
```

On Windows for local smoke tests:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## Train

T4 first run:

```bash
python -m src.train --config configs/t4_mistral_alpaca_qlora.json
```

A100 larger run:

```bash
python -m src.train --config configs/a100_mistral_openhermes_qlora.json
```

## Evaluate

Compute held-out perplexity:

```bash
python -m src.evaluate \
  --config configs/t4_mistral_alpaca_qlora.json \
  --adapter-dir outputs/mistral7b-alpaca-cleaned-qlora-r8 \
  --max-eval-samples 500 \
  --output reports/eval_adapter.json
```

For MMLU, install `lm-eval` from `requirements.txt` and run:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=mistralai/Mistral-7B-v0.3,peft=outputs/mistral7b-alpaca-cleaned-qlora-r8,load_in_4bit=True \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path reports/mmlu_adapter
```

Use MMLU only as a secondary signal for Alpaca. Instruction tuning may improve helpfulness and formatting without improving factual multiple-choice performance.

## Run Ablations

Preview commands:

```powershell
python -m src.ablate --grid configs/ablation_grid.json --dry-run
```

Run the grid:

```bash
python -m src.ablate --grid configs/ablation_grid.json
```

Minimum credible ablation set:

- Rank: `4`, `8`, `16`, `32`
- Target modules: `q+v`, all attention projections, all linear layers
- Alpha: `r`, `2r`, `32`
- Learning rate: `1e-4`, `2e-4`, `5e-4`
- QLoRA vs LoRA, only if you have an A100 or equivalent memory

## Inference

Adapter-only inference:

```bash
python -m src.infer \
  --config configs/t4_mistral_alpaca_qlora.json \
  --adapter-dir outputs/mistral7b-alpaca-cleaned-qlora-r8 \
  --prompt "### Instruction:\nExplain LoRA in two paragraphs.\n\n### Response:\n"
```

Merged models are simpler to serve, but adapter-only loading is better when you want one base model with multiple task adapters. KV cache works normally because LoRA changes the projection weights used to compute keys and values; the cache stores the resulting activations during generation.

## What To Measure

Record these for every run:

- Base validation perplexity.
- Adapter validation perplexity.
- Trainable parameter count and percentage.
- Peak VRAM from `nvidia-smi`.
- Wall-clock time.
- MMLU or task benchmark score.
- At least 10 fixed qualitative prompts with base and adapter outputs.

## Resume-Worthy Results

Strong results an interviewer will respect:

- You fit Mistral 7B on a single T4 using NF4 QLoRA under 16GB VRAM.
- You report exact trainable parameters, not just "used LoRA".
- You show a base-vs-adapter held-out perplexity improvement.
- You run rank and target-module ablations and can explain the tradeoff.
- You know when fine-tuning did not help and can diagnose data quality, benchmark mismatch, or overfitting.

Measured bullets for the current in-progress run:

- Fine-tuned Mistral-7B-v0.3 with rank-8 NF4 QLoRA on Alpaca-cleaned,
  training 6.82M adapter parameters across `q_proj`, `k_proj`, `v_proj`, and
  `o_proj` while keeping the 7B-class base model frozen.
- Resumed the main single-GPU run from checkpoint 600 and progressed past
  checkpoint 900 / 1,200 target optimizer steps; latest observed validation
  loss improved from 1.0143 early in training to 0.9648 during the resumed run.
- Implemented manual LoRA from first principles in PyTorch plus production
  QLoRA with PEFT, bitsandbytes NF4, double quantization, gradient
  checkpointing, and paged AdamW.
- Final resume bullets should add final validation perplexity, peak VRAM,
  qualitative prompt results, and any ablation results only after evaluation is
  complete.

Current status docs:

- [TRAINING_STATUS.md](reports/TRAINING_STATUS.md)
- [ATS_SCREENING_PACK.md](reports/ATS_SCREENING_PACK.md)

## Interview Depth Checklist

Be ready to answer:

- Why does LoRA use `B @ A` instead of training the full delta matrix?
- What does `alpha / r` do, and why can too-large alpha destabilize training?
- Why do `q_proj` and `v_proj` often matter most?
- Why can instruction tuning reduce perplexity but not improve MMLU?
- What memory is saved by QLoRA, and what memory remains from activations and optimizer states?
- Why does low-quality synthetic data sometimes make the model worse than the base model?
- What changed when you increased rank from 8 to 32?

Surface-level red flags:

- No held-out validation set.
- Only qualitative examples.
- No parameter count.
- No base-model comparison.
- No ablations.
- Copy-pasted training script with no explanation of LoRA math or memory.

## Common Mistakes

1. Training on assistant responses without masking or consistent formatting, causing prompt leakage or bad chat behavior.
2. Forgetting `prepare_model_for_kbit_training` before QLoRA.
3. Using too-long sequence lengths on T4 and blaming LoRA for OOM.
4. Evaluating only on training-like prompts.
5. Reporting benchmark gains without a base-model baseline under the same evaluation pipeline.

Overfitting signs:

- Training loss keeps dropping while validation perplexity rises.
- Qualitative outputs memorize dataset style or become verbose boilerplate.
- Rank 32 improves train loss but not validation metrics.

The honest framing for null results is: "Instruction tuning improved output format and reduced held-out Alpaca perplexity, but did not improve MMLU, suggesting the dataset improved instruction-following style rather than factual reasoning."
