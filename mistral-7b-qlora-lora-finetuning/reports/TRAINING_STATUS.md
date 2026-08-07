# Training Status

Last checked: 2026-08-08 03:05 IST.

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
| Latest observed checkpoint | `checkpoint-900` |
| Active process | resumed from `checkpoint-600` |
| Latest observed progress | around 978 / 1,200 steps from stderr tail |
| Latest observed eval loss | 0.964837 during resumed run |
| Verification | `python -m pytest -q` passed, 2 tests |

## Active Command

```powershell
D:\venvs\qlora-ft2\Scripts\python.exe -u -m src.train --config configs/t4_mistral_alpaca_qlora.json --resume-from-checkpoint outputs/mistral7b-alpaca-cleaned-qlora-r8/checkpoint-600
```

## Completion Criteria

- Let training reach 1,200 steps without interrupting the process.
- Run held-out perplexity evaluation on the final adapter.
- Run at least 10 fixed qualitative prompts and save base-vs-adapter examples.
- Record peak VRAM from `nvidia-smi`.
- Update resume bullets with final measured metrics only.

## Final Evaluation Command

```powershell
python -m src.evaluate `
  --config configs/t4_mistral_alpaca_qlora.json `
  --adapter-dir outputs/mistral7b-alpaca-cleaned-qlora-r8 `
  --max-eval-samples 500 `
  --output reports/eval_adapter.json
```
