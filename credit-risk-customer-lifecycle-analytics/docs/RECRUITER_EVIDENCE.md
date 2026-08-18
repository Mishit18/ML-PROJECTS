# Recruiter Evidence

## Project

Credit Risk Scorecard and Portfolio Analytics

## Data provenance

Home Credit public application and linked behavioral tables.

## Truth boundary

Real public data; approval/loss economics are explicitly modeled assumptions.

## Primary evidence

- `reports/home_credit_real_data_validation.md`
- `reports/model_card.md`
- `docs/interview_defense.md`

## Verification

1. Create the environment using the repository dependency specification.
2. Run the test command documented in the README.
3. Run the evidence-generation command documented in the README when compute and data access permit.
4. Compare regenerated outputs with the primary evidence above.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
