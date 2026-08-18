# Recruiter Evidence

## Project

ML Serving and Drift Monitoring

## Data provenance

307,511 Home Credit applications and 17.0M linked behavioral records.

## Truth boundary

Real public data; local API load test; no claim of internet-scale production traffic.

## Primary evidence

- `reports/model_card.md`
- `reports/load_benchmark.json`
- `reports/governance_readiness.json`

## Verification

1. Create the environment using the repository dependency specification.
2. Run the test command documented in the README.
3. Run the evidence-generation command documented in the README when compute and data access permit.
4. Compare regenerated outputs with the primary evidence above.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
