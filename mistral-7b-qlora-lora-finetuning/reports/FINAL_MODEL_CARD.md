# Final Model Card: Mistral-7B QLoRA Adapter

## Model and Data

| Item | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-v0.3` |
| Dataset | `yahma/alpaca-cleaned` |
| Method | QLoRA with NF4 and double quantization |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | 6,815,744 |
| Trainable percent | 0.0939% |
| Max sequence length | 1024 |
| Optimizer | paged AdamW 8-bit |
| Gradient accumulation | 16 |
| Max steps | 1,200 |

## Final Evidence

The run completed the planned 1,200 training steps and saved `checkpoint-1200`.

| Metric | Value |
|---|---:|
| First logged eval loss, step 100 | 1.014257 |
| Final logged eval loss, step 1200 | 0.962782 |
| Relative eval-loss improvement | 5.07% |
| Eval throughput near final checkpoint | 4.214 samples/sec |
| Base held-out eval loss, 1,000 samples | 1.602915 |
| Adapter held-out eval loss, 1,000 samples | 0.962810 |
| Base held-out perplexity, 1,000 samples | 4.9675 |
| Adapter held-out perplexity, 1,000 samples | 2.6190 |
| Relative perplexity reduction vs base | 47.27% |
| Qualitative prompt regression | 5 deterministic base-vs-adapter prompts saved |
| Rank-4 small ablation | 3.41M trainable params, 100 steps, eval loss 1.0312 |

## Resume-Safe Claim

Fine-tuned Mistral-7B-v0.3 with rank-8 NF4 QLoRA on Alpaca-cleaned, training 6.82M adapter parameters while freezing the 7B-class base model; completed a 1,200-step single-GPU run and reduced held-out validation perplexity from 4.97 to 2.62 on 1,000 samples.

## Do Not Claim Yet

- Do not claim MMLU improvement until `lm_eval` is run.
- Do not claim hallucination reduction; fixed qualitative prompts still show factual and overclaiming failures.
- Do not claim production readiness without latency, safety, and serving tests.
- Do not claim full fine-tuning equivalence.

## Next Evidence Layer

1. Expand prompt regression from 5 to 10-20 prompts and add pass/fail rubrics.
2. Run rank 8 and rank 16 under the same 100-step ablation budget used for the completed rank-4 small run.
3. Record peak VRAM and wall-clock training time.
