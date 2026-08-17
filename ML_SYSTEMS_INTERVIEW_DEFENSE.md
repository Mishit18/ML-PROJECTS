# ML Systems Interview Defense

## Portfolio Story

The ML resume should be defended as a complete ML stack:

1. **Modeling from first principles**: Mini-GPT and DDPM show core architecture understanding.
2. **Large-model adaptation**: Mistral-7B QLoRA shows parameter-efficient fine-tuning under memory constraints.
3. **Production discipline**: Real-Time ML Serving shows FastAPI inference, model cards, monitoring, latency telemetry, Docker, and tests.
4. **Applied business ML**: FlowFinance and credit-risk analytics show LLM routing, categorization, and governance-oriented analytics.

## QLoRA Talking Points

- LoRA trains low-rank adapter matrices while freezing the base model.
- QLoRA stores the base model in 4-bit NF4, trains adapters, and uses paged optimizers to reduce memory spikes.
- The completed run trained 6.82M adapter parameters, only 0.0939% of the base parameter count.
- The validation loss improved from 1.0143 at step 100 to 0.9628 at step 1200.

## DDPM Talking Points

- The model predicts noise epsilon rather than directly predicting clean images.
- The forward noising process has a closed form, so noisy samples can be drawn at arbitrary timesteps.
- DDIM sampling trades stochasticity for faster deterministic sampling.
- EMA weights usually improve generation stability because they smooth noisy parameter updates.
- Final evaluation used 50,000 generated samples with DDIM-50 and reported FID 10.0958, Inception Score 8.7801 +/- 0.0958, and around 10 samples/sec.

## Mini-GPT Talking Points

- The model uses a modern decoder-only architecture with causal attention, RoPE, RMSNorm, SwiGLU, grouped-query attention, tied embeddings, and KV-cache decoding.
- The strongest evidence is measured improvement, not just implementation: validation perplexity improved from 14.25 to 9.82 after longer training and low-LR continuation.
- The old GPT-style baseline comparison is a controlled small-model-budget comparison, so present it as a relative architecture/training improvement rather than a claim of frontier LLM quality.

## Serving Talking Points

- The serving project separates model artifact loading, request validation, API endpoints, logging, latency metrics, and drift checks.
- PSI drift is a lightweight first warning system, not automated retraining.
- Model cards document metrics, assumptions, limitations, and intended use.

## What Not To Overclaim

- QLoRA is completed, but MMLU and fixed-prompt human eval are not yet final evidence.
- DDPM has final FID/IS on CIFAR-10, but it is still a class-unconditional academic image-generation project, not a production image model.
- FastAPI serving is production-style, not a deployed commercial system.
