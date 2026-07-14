import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999).float()


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.timesteps = timesteps
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_log_variance_clipped", torch.log(self.posterior_variance))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def p_mean_variance(self, model: nn.Module, x_t: torch.Tensor, timesteps: torch.Tensor):
        pred_noise = model(x_t, timesteps)
        x0 = self.predict_start_from_noise(x_t, timesteps, pred_noise).clamp(-1.0, 1.0)
        mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x0
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        log_variance = extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return mean, log_variance, x0

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        mean, log_variance, _ = self.p_mean_variance(model, x_t, timesteps)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def sample_ddpm(self, model: nn.Module, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        model.eval()
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        model.eval()
        times = torch.linspace(-1, self.timesteps - 1, steps + 1, device=device).long()
        times = list(reversed(times.tolist()))
        x = torch.randn(shape, device=device)
        for time, time_next in zip(times[:-1], times[1:]):
            t = torch.full((shape[0],), time, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            alpha = self.alphas_cumprod[time]
            alpha_next = torch.tensor(1.0, device=device) if time_next < 0 else self.alphas_cumprod[time_next]
            x0 = ((x - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()).clamp(-1.0, 1.0)
            sigma = eta * (((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).clamp(min=0).sqrt())
            c = (1 - alpha_next - sigma**2).clamp(min=0).sqrt()
            noise = torch.randn_like(x) if time_next >= 0 else torch.zeros_like(x)
            x = alpha_next.sqrt() * x0 + c * pred_noise + sigma * noise
        return x

    def training_losses(self, model: nn.Module, x_start: torch.Tensor) -> torch.Tensor:
        batch = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        pred_noise = model(x_t, t)
        return F.mse_loss(pred_noise, noise)
