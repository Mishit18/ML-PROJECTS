# Recruiter Evidence

## Project

Mistral-7B QLoRA Instruction Tuning

## Data provenance

Alpaca-cleaned instruction data plus held-out evaluation and ARC-Easy diagnostic.

## Truth boundary

Real single-GPU fine-tuning and evaluation; downstream benchmark result is mixed and documented.

## Primary evidence

- `reports/FINAL_REPORT.md`
- `reports/FINAL_MODEL_CARD.md`
- `reports/ARC_EASY_DIAGNOSTIC.md`

## Verification

1. Create the environment using the repository dependency specification.
2. Run the test command documented in the README.
3. Run the evidence-generation command documented in the README when compute and data access permit.
4. Compare regenerated outputs with the primary evidence above.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
