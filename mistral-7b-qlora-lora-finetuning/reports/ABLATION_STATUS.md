# Ablation Status

## Completed Evidence

- Main rank-8 QLoRA run completed 1,200 steps.
- Main adapter reduced held-out perplexity from 4.97 to 2.62 on 1,000 validation samples.
- Five deterministic base-vs-adapter qualitative prompts are saved in `reports/generation_comparison.json`.

## Completed Small Ablation

| Run | Status | Resume-Safe? | Notes |
|---|---|---|---|
| `configs/ablate_rank4_100steps.json` | Completed 100 steps | Limited | Rank-4 adapter trained 3.41M parameters and reached eval loss 1.0312 after 100 steps; use only as a small-run ablation, not as a final model comparison. |

## Next Valid Ablation Plan

Run rank 8 and rank 16 under the same 100-step budget, then compare eval loss, trainable parameters, and wall-clock time under identical settings.
