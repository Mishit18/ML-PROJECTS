# Training Status

Last checked: 2026-08-08 04:10 IST.

## Active Run

| Item | Value |
| --- | --- |
| Model | `mistralai/Mistral-7B-v0.3` |
| Dataset | `yahma/alpaca-cleaned` |
| Method | QLoRA, NF4, double quantization |
| LoRA rank / alpha | rank 8, alpha 16 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable params | 6,815,744 |
| Reported trainable percent | 0.0939% of full base model params |
| Config | `configs/t4_mistral_alpaca_qlora.json` |
| Output dir | `outputs/mistral7b-alpaca-cleaned-qlora-r8` |
| Target steps | 1,200 |
| Latest observed checkpoint | `checkpoint-1200` |
| Active process | completed 1,200 / 1,200 steps |
| Latest observed progress | final checkpoint saved |
| Latest observed eval loss | 0.962782 at step 1,200 |
| Verification | `python -m pytest -q` passed, 2 tests; base/adapted held-out eval completed |

## Completed Training Command

```powershell
D:\venvs\qlora-ft2\Scripts\python.exe -u -m src.train --config configs/t4_mistral_alpaca_qlora.json --resume-from-checkpoint outputs/mistral7b-alpaca-cleaned-qlora-r8/checkpoint-600
```

## Completion Criteria

- Training reached 1,200 steps and saved final adapter/checkpoint.
- Held-out evaluation completed on the same 1,000-sample split for base and adapter.
- Run at least 10 fixed qualitative prompts and save base-vs-adapter examples.
- Record peak VRAM from `nvidia-smi`.
- Update resume bullets with final measured metrics only.

## Held-Out Base vs Adapter Evaluation

| Model | Eval Loss | Perplexity | Eval Samples |
|---|---:|---:|---:|
| Base Mistral-7B-v0.3 | 1.602915 | 4.9675 | 1,000 |
| Rank-8 QLoRA adapter | 0.962810 | 2.6190 | 1,000 |

The final adapter reduced held-out perplexity by 47.27% versus the frozen base model under the same evaluation script and sample count.

## Evaluation Loss Curve

| Step | Eval Loss |
|---:|---:|
| 100 | 1.014257 |
| 200 | 0.996753 |
| 300 | 0.988627 |
| 400 | 0.983328 |
| 500 | 0.977995 |
| 600 | 0.974084 |
| 700 | 0.969701 |
| 800 | 0.967229 |
| 900 | 0.964837 |
| 1000 | 0.963409 |
| 1100 | 0.962931 |
| 1200 | 0.962782 |

Relative improvement from the first logged evaluation to the final logged evaluation: 5.07%.

## Final Evaluation Command

```powershell
D:\venvs\qlora-ft2\Scripts\python.exe -u -m src.evaluate `
  --config configs/t4_mistral_alpaca_qlora.json `
  --adapter-dir outputs/mistral7b-alpaca-cleaned-qlora-r8 `
  --max-eval-samples 1000 `
  --output reports/eval_adapter_1000.json
```
