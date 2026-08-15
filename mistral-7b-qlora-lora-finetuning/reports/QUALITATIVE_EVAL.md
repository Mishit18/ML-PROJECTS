# Qualitative Evaluation: Base vs QLoRA Adapter

## Setup

| Item | Value |
|---|---|
| Prompts | 5 fixed instruction prompts from `reports/sample_prompts.json` |
| Decoding | Greedy / deterministic |
| Max new tokens | 96 |
| Base output | `reports/generation_base.json` |
| Adapter output | `reports/generation_adapter.json` |
| Side-by-side artifact | `reports/generation_comparison.json` |

## Findings

- The adapter follows the requested instruction format more consistently than the base model on LoRA explanation, CUDA OOM debugging, and validation-perplexity diagnosis prompts.
- The base model gives a clearly wrong LoRA equation and repeats generic process-monitoring steps in the CUDA OOM prompt.
- The adapter still produces unsafe resume-style overclaims on the resume-bullet prompt and incomplete/incorrect rank-parameter details on the LoRA-rank prompt.
- The qualitative result supports a conservative claim: instruction tuning improved task format and reduced held-out perplexity, but it does not prove factual-reasoning or benchmark improvement.

## Interview-Safe Interpretation

This project should be presented as a reproducible parameter-efficient fine-tuning and evaluation pipeline, not as a production chatbot. The strongest evidence is the 1,000-sample held-out perplexity reduction from 4.97 to 2.62. The qualitative prompts are useful as regression tests and reveal the next work items: factuality checks, constrained resume-claim generation, and MMLU or domain benchmark evaluation.
