# Mini-GPT Baseline And Ablation Report

All rows are generated from checkpoint summaries. `pending` means the run/config exists but has not been trained and summarized yet.

## Baseline vs Modern Decoder

| Run | Dataset | Tokens trained | Train loss | Val loss | Perplexity | Architecture |
|---|---|---:|---:|---:|---:|---|
| Baseline GPT block | tinystories | 5092585 | 2.9074 | 2.9700 | 19.49 | Learned positions + LayerNorm + GELU + full MHA |
| Modern decoder | tinystories | 5092585 | 2.7068 | 2.6565 | 14.25 | RoPE + RMSNorm + SwiGLU + GQA |
| Modern 20M tokens | tinystories | 15277755 | 1.9133 | 2.3053 | 10.03 | Same modern decoder, longer TinyStories run |

Modern decoder reduced validation perplexity from 19.49 to 14.25 at the same small-model parameter budget (26.9% relative reduction).

## Focused Ablations

| Ablation | Dataset | Tokens trained | Train loss | Val loss | Perplexity | Changed feature |
|---|---|---:|---:|---:|---:|---|
| RoPE vs learned positions | tinystories | 3055551 | 3.4548 | 3.2213 | 25.06 | learned absolute positions |
| SwiGLU vs GELU | tinystories | 3055551 | 3.3489 | 3.0641 | 21.41 | GELU FFN |
| GQA vs full MHA | tinystories | 3055551 | 3.3848 | 2.9548 | 19.20 | full MHA |
