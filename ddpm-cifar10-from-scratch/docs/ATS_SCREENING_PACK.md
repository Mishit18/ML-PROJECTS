# ATS Screening Pack

## Best-Fit Roles

- Machine Learning Engineer Intern
- AI/ML Research Intern
- Generative AI Intern
- Computer Vision Intern
- Data Scientist, Deep Learning

## Strong Resume Bullets

- Implemented a DDPM for CIFAR-10 from scratch in PyTorch, including forward
  noising, epsilon-prediction denoising, cosine/linear beta schedules, EMA
  checkpoints, DDPM ancestral sampling, and deterministic DDIM sampling.
- Built a 71.03M-parameter residual U-Net with sinusoidal timestep embeddings,
  GroupNorm/SiLU residual blocks, skip-connected encoder-decoder stages, and
  multi-head attention at 16x16 and 8x8 resolutions.
- Added training infrastructure with mixed precision, TF32, gradient
  accumulation, warmup + cosine LR decay, gradient clipping, resumable
  checkpoints, sample grids, monitoring, and pytest coverage.
- Completed 300,000 optimizer steps and evaluated 50,000 EMA DDIM-50 samples,
  achieving FID 10.0958 and Inception Score 8.7801 +/- 0.0958 at approximately
  10 samples per second.

## ATS Keywords

DDPM, Denoising Diffusion Probabilistic Model, Diffusion Models, Generative AI,
CIFAR-10, U-Net, Residual U-Net, Sinusoidal Timestep Embeddings, Self-Attention,
GroupNorm, SiLU, EMA, DDIM, FID, Inception Score, Mixed Precision, AMP, TF32,
PyTorch, Computer Vision, Deep Learning, Generative Modeling.

## Claims To Avoid

- Report final metrics with the sample count, EMA weights, DDIM sampler, and 50 denoising steps.
- Do not claim state-of-the-art or production image quality.
- Do not claim the model uses `diffusers`; the point is from-scratch PyTorch.
- Do not imply the remaining raw-weight or sampler ablations are complete.

## Upgrade Path To 100/100

- Complete the 300k-step run.
- Evaluate EMA and raw checkpoints with 50,000 generated samples.
- Benchmark DDIM 25/50/100 against DDPM 1000.
- Add ablations for cosine vs linear schedule, EMA vs raw, and model size.
- Add a final gallery with fixed seeds across checkpoints.
- Add a concise model card covering data, limitations, compute, and metrics.
