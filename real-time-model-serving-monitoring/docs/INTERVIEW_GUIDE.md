# Interview Guide

## Two-minute explanation

**Problem (20 seconds):** ML Serving and Drift Monitoring addresses a concrete modeling or decision problem rather than demonstrating an algorithm in isolation.

**Data and controls (25 seconds):** The study uses 307,511 Home Credit applications and 17.0M linked behavioral records. I separated train/calibration/holdout or scenario-generation stages as appropriate and documented provenance and leakage controls.

**Method (35 seconds):** The pipeline follows Home Credit tables -> Leakage-safe features -> LightGBM + calibration -> FastAPI contract -> Monitoring + governance. I compared against explicit baselines and retained failure cases instead of reporting only favorable outputs.

**Result (25 seconds):** I report the exact metrics in the committed evidence artifacts. The defensible boundary is: Real public data; local API load test; no claim of internet-scale production traffic.

**Limitations and next step (15 seconds):** The next step is external or forward validation under the real operating constraints documented in the repository.

## Ten difficult questions

1. What exact decision does ML Serving and Drift Monitoring support, and who would act on its output?
2. Which parts use 307,511 Home Credit applications and 17.0M linked behavioral records, and where could leakage or look-ahead bias enter?
3. Why were the selected baselines appropriate, and which stronger baseline would you add next?
4. Why is the headline metric decision-relevant, and what complementary metric could reverse the conclusion?
5. Which assumption contributes the most model risk, and how did you stress it?
6. What failed during development, and what evidence caused you to change or reject an approach?
7. How can another reviewer reproduce the result from a clean environment without private knowledge?
8. What breaks first under scale, latency, distribution shift, or adversarial inputs?
9. Which result is real, simulated, modeled, or estimated, and why is that distinction important?
10. Open the primary evidence artifact and derive one resume metric from the underlying output.

## Evidence to open during an interview

- `reports/model_card.md`
- `reports/load_benchmark.json`
- `reports/governance_readiness.json`
