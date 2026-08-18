# Interview Guide

## Two-minute explanation

**Problem (20 seconds):** CIFAR-10 DDPM Image Generation addresses a concrete modeling or decision problem rather than demonstrating an algorithm in isolation.

**Data and controls (25 seconds):** The study uses CIFAR-10 training set and 50,000 generated evaluation samples. I separated train/calibration/holdout or scenario-generation stages as appropriate and documented provenance and leakage controls.

**Method (35 seconds):** The pipeline follows CIFAR-10 -> Cosine noising -> U-Net epsilon model -> EMA checkpoints -> DDIM sampling + FID/IS. I compared against explicit baselines and retained failure cases instead of reporting only favorable outputs.

**Result (25 seconds):** I report the exact metrics in the committed evidence artifacts. The defensible boundary is: Real model training and generation evaluation; throughput measured locally.

**Limitations and next step (15 seconds):** The next step is external or forward validation under the real operating constraints documented in the repository.

## Ten difficult questions

1. What exact decision does CIFAR-10 DDPM Image Generation support, and who would act on its output?
2. Which parts use CIFAR-10 training set and 50,000 generated evaluation samples, and where could leakage or look-ahead bias enter?
3. Why were the selected baselines appropriate, and which stronger baseline would you add next?
4. Why is the headline metric decision-relevant, and what complementary metric could reverse the conclusion?
5. Which assumption contributes the most model risk, and how did you stress it?
6. What failed during development, and what evidence caused you to change or reject an approach?
7. How can another reviewer reproduce the result from a clean environment without private knowledge?
8. What breaks first under scale, latency, distribution shift, or adversarial inputs?
9. Which result is real, simulated, modeled, or estimated, and why is that distinction important?
10. Open the primary evidence artifact and derive one resume metric from the underlying output.

## Evidence to open during an interview

- `docs/EXPERIMENT_REPORT.md`
- `results/FINAL_TRAINING_SUMMARY.md`
- `results/sampler_speed_benchmark.json`
