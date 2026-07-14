# DDPM From Scratch on CIFAR-10

PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) for unconditional CIFAR-10 generation. The diffusion algorithm is implemented from PyTorch primitives only: no `diffusers`, no `guided-diffusion`.

## Project Highlights

- Implements the DDPM forward process, learned reverse process, and epsilon-prediction objective from scratch.
- Uses a residual U-Net with sinusoidal timestep conditioning, GroupNorm/SiLU blocks, skip connections, and multi-head self-attention.
- Supports linear and cosine schedules, DDPM ancestral sampling, and deterministic DDIM sampling.
- Includes EMA weights, mixed precision training, TF32 acceleration, gradient accumulation, warmup + cosine LR decay, and resumable checkpoints.
- Logs training curves, sample grids, FID, Inception Score, sampler speed, and ablation metadata.

## Quick Start

```powershell
cd "D:\GITHUB\ML PROJECTS\ddpm-cifar10-from-scratch"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --config configs/cifar10_rtx4060_best.yaml
```

Monitor the active run:

```powershell
python monitor.py --run-dir runs/cifar10_rtx4060_best
```

Generate samples from the latest checkpoint:

```powershell
python sample.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --num-samples 64
python sample.py --run-dir runs/cifar10_rtx4060_best --sampler ddpm --num-samples 64
```

Compute FID and Inception Score:

```powershell
python evaluate.py --run-dir runs/cifar10_rtx4060_best --sampler ddim --ddim-steps 50 --num-samples 50000
```

Metrics are written to `runs/<run_name>/metrics/metrics.csv` and `runs/<run_name>/metrics/*.json`.

## 1. U-Net Architecture

The implementation is in `src/ddpm/unet.py`.

Recommended CIFAR-10 model:

| Variant | Base channels | Channel multipliers | Resolutions | Attention | Approx use |
| --- | ---: | --- | --- | --- | --- |
| Small | 64 | `[1, 2, 2, 4]` | 32, 16, 8, 4 | 16 | T4 initial results |
| Medium | 128 | `[1, 2, 2, 4]` | 32, 16, 8, 4 | 16, 8 | A100 or longer T4 run |
| RTX4060 best | 128 | `[1, 2, 2, 4]` | 32, 16, 8, 4 | 16, 8 | Main local run |

Each resolution level has 2 residual blocks. The encoder starts at 32x32, then downsamples to 16x16, 8x8, and 4x4. The decoder mirrors the encoder and concatenates skip activations from matching encoder stages.

Residual block:

1. `GroupNorm -> SiLU -> Conv2d`
2. Add projected timestep embedding as a per-channel bias
3. `GroupNorm -> SiLU -> Dropout -> Conv2d`
4. Residual skip through identity or `1x1 Conv2d`

Attention block:

- Multi-head self-attention over flattened spatial tokens.
- At 16x16 for the small model.
- At 16x16 and 8x8 for the medium model.
- Avoid 32x32 attention initially because 1024 spatial tokens are expensive.

Timestep embedding:

```python
emb[t, 2i]   = cos(t / 10000^(2i / dim))
emb[t, 2i+1] = sin(t / 10000^(2i / dim))
```

Then an MLP maps `base_channels -> 4 * base_channels -> 4 * base_channels`. Each residual block projects this vector to its output channel count and adds it to the feature map:

```python
h = h + time_proj(silu(t_emb))[:, :, None, None]
```

## 2. Forward Process

The forward diffusion process gradually corrupts clean data `x_0`:

```text
q(x_t | x_{t-1}) = N(sqrt(alpha_t) x_{t-1}, beta_t I)
alpha_t = 1 - beta_t
alpha_bar_t = product_{s=1}^t alpha_s
```

Because Gaussian transitions compose, we can sample `x_t` in one step:

```text
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
epsilon ~ N(0, I)
```

The code precomputes:

- `betas`
- `alphas = 1 - betas`
- `alphas_cumprod = alpha_bar`
- `sqrt_alphas_cumprod`
- `sqrt_one_minus_alphas_cumprod`
- posterior coefficients used by DDPM sampling

Schedules implemented in `src/ddpm/diffusion.py`:

- Linear: `beta_t` linearly increases from `1e-4` to `0.02`.
- Cosine: Nichol & Dhariwal style cumulative alpha schedule:

```text
alpha_bar_t = cos^2(((t / T + s) / (1 + s)) * pi / 2) / cos^2((s / (1 + s)) * pi / 2)
```

## 3. Reverse Process and Objective

The model predicts noise:

```text
epsilon_theta(x_t, t) ~= epsilon
L_simple = E[ || epsilon - epsilon_theta(x_t, t) ||_2^2 ]
```

Why predict epsilon?

- It is the standard simple objective from DDPM.
- It is numerically stable.
- It avoids forcing the network to directly reconstruct clean pixels from very noisy states.
- It works well on CIFAR-10 without extra weighting.

Other parameterizations:

- Predict `x_0`: direct denoised image prediction; can be unstable at high noise.
- Predict `v`: used in many modern systems; often more balanced across noise levels, but epsilon prediction is the cleanest first implementation.

## 4. Sampling

DDPM ancestral sampling uses all 1000 reverse steps:

```text
p_theta(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 I)
```

The code predicts `epsilon`, reconstructs `x_0`, then uses the closed-form posterior mean:

```text
x0_hat = (x_t - sqrt(1 - alpha_bar_t) epsilon_theta) / sqrt(alpha_bar_t)
mu = c1 * x0_hat + c2 * x_t
```

DDIM sampling skips timesteps by treating the reverse path as a non-Markovian process with the same training objective. For a selected sequence `tau`, it jumps from `t` to `t_next`:

```text
x_{t_next} =
  sqrt(alpha_bar_{t_next}) x0_hat
  + sqrt(1 - alpha_bar_{t_next} - sigma_t^2) epsilon_theta
  + sigma_t z
```

With `eta = 0`, `sigma_t = 0`, so sampling is deterministic. This allows 50 to 100 steps instead of 1000, usually giving a large speedup with modest FID loss.

## 5. Training Setup

Use CIFAR-10 normalized to `[-1, 1]`. Augmentation: random horizontal flip only.

Small T4-friendly setup:

- Batch size: 128
- Optimizer: AdamW
- Learning rate: `2e-4`
- Weight decay: `0.01`
- Gradient clipping: `1.0`
- EMA decay: `0.9999`
- Mixed precision: `torch.cuda.amp`
- Epochs for initial samples: 50 to 100
- Better run: 200+ epochs or 400k+ optimizer steps

A T4 under 6 hours should give recognizable samples, but do not expect benchmark-level FID from a short run. A100 or longer T4 training is better for final resume numbers.

Main RTX 4060 setup:

- Config: `configs/cifar10_rtx4060_best.yaml`
- Parameters: about 71M
- Batch size: 64
- Gradient accumulation: 2
- Effective batch size: 128
- Training budget: 300k optimizer steps
- Optimizer: AdamW, LR `2e-4`
- LR schedule: 2k-step warmup, cosine decay to 5% of peak LR
- Precision: AMP + TF32

## 6. Evaluation

FID measures the Frechet distance between Inception feature distributions of real and generated images. Lower is better. It is sensitive to sample count, preprocessing, and implementation.

Inception Score measures whether generated images are classifiable and diverse. Higher is better, but it can miss artifacts and should not replace FID.

Protocol:

1. Train with EMA.
2. Generate 50,000 samples.
3. Compare to CIFAR-10 test images using `torchmetrics`.
4. Report sampler, steps, checkpoint, number of generated samples, and seed.

Rough CIFAR-10 expectations for a from-scratch student project:

- FID 40-80: recognizable early baseline.
- FID 20-40: solid small-model result.
- FID 10-20: strong result for a compact U-Net.
- Below 10: excellent and likely needs more compute, tuning, or a larger model.

## 7. Ablations

Run these as separate configs and record FID, IS, sampling time, parameter count, and training time.

| Ablation | Compare | Expected result |
| --- | --- | --- |
| Schedule | linear vs cosine | Cosine often improves FID and training stability |
| Sampler | DDPM 1000 vs DDIM 50 | DDIM much faster, usually slightly worse FID |
| Model size | small vs medium | Medium improves quality but costs memory/time |
| EMA | EMA vs raw weights | EMA usually improves visual quality and FID |

Use one CSV table:

```text
run,params_m,schedule,sampler,steps,ema,fid,is_mean,is_std,samples_per_sec,train_hours
```

Ablation assets:

- `configs/abl_cifar10_linear_rtx4060.yaml`: same medium U-Net with a linear schedule.
- `configs/abl_cifar10_small_cosine.yaml`: smaller U-Net for model-size ablation.
- `evaluate.py --weights ema` vs `evaluate.py --weights raw`: EMA ablation from the same checkpoint.
- `benchmark_sampler.py`: DDIM/DDPM sampler speed comparison.
- `results/ablation_plan.csv`: experiment checklist.

## 8. Code Structure

```text
ddpm-cifar10-from-scratch/
  configs/
    cifar10_small.yaml
    cifar10_medium.yaml
  src/ddpm/
    unet.py       # U-Net, ResBlock, attention, timestep embeddings
    diffusion.py  # schedules, q_sample, DDPM and DDIM sampling
    ema.py        # exponential moving average weights
    data.py       # CIFAR-10 loader and transforms
    utils.py      # config, seeding, image saving
  train.py        # training loop
  sample.py       # DDPM/DDIM sample generation
  evaluate.py     # FID and Inception Score
  monitor.py      # training status utility
  benchmark_sampler.py
  docs/
    EXPERIMENT_REPORT.md
    TROUBLESHOOTING.md
  results/
    ablation_plan.csv
```

Log during training:

- Training MSE loss
- Learning rate
- Wall-clock time
- Sample grids every few epochs
- Checkpoint FID/IS after meaningful intervals
- Sampling speed for DDPM and DDIM

## 9. Resume Bullet Templates

Fill these only after real experiments:

- Implemented DDPM from scratch in PyTorch, including closed-form forward noising, learned reverse denoising U-Net, cosine noise schedule, and EMA sampling; achieved FID `[X]` on CIFAR-10 with 50k generated samples.
- Built a `[Z]`M-parameter U-Net with `[X]` residual blocks, GroupNorm/SiLU residual pathways, sinusoidal timestep conditioning, and multi-head attention at `[Y]` resolutions; trained with mixed precision on a single GPU.
- Implemented deterministic DDIM sampling, reducing generation from 1000 denoising steps to `[S]` steps for `[X]`x inference speedup at matched sample quality, with FID delta `< [Y]`.
- Ablated linear vs cosine schedules, EMA vs non-EMA, model size, and DDPM vs DDIM; cosine reduced FID by `[X]%` and EMA improved FID by `[Y]%` over the non-EMA baseline.

## 10. Common Mistakes

1. Wrong image scale: training on `[0, 1]` while sampling assumes `[-1, 1]`.
2. Incorrect timestep indexing: using `alpha_bar[t-1]` where `alpha_bar[t]` is required.
3. Forgetting to stop noise at `t = 0` during DDPM sampling.
4. Broken skip connections: concatenating encoder features with mismatched resolutions or channels.
5. Evaluating raw model weights instead of EMA weights, causing worse samples and noisy FID.

## Suggested Milestone Plan

1. Overfit-check on one batch: loss should decrease clearly.
2. Train small cosine model for 50-100 epochs and inspect DDIM grids.
3. Compute FID with 10k generated samples for fast iteration.
4. Run final FID with 50k generated samples.
5. Run ablations and fill the resume bullets with actual measured values.
