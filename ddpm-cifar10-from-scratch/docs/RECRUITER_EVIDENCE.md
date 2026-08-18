# Recruiter Evidence

## Project

CIFAR-10 DDPM Image Generation

## Data provenance

CIFAR-10 training set and 50,000 generated evaluation samples.

## Truth boundary

Real model training and generation evaluation; throughput measured locally.

## Primary evidence

- `docs/EXPERIMENT_REPORT.md`
- `results/FINAL_TRAINING_SUMMARY.md`
- `results/sampler_speed_benchmark.json`

## Verification

1. Create the environment using the repository dependency specification.
2. Run the test command documented in the README.
3. Run the evidence-generation command documented in the README when compute and data access permit.
4. Compare regenerated outputs with the primary evidence above.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
