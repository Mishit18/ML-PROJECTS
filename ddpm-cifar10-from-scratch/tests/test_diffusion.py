import torch

from src.ddpm.diffusion import GaussianDiffusion


class ZeroModel(torch.nn.Module):
    def forward(self, x, timesteps):
        return torch.zeros_like(x)


def test_q_sample_closed_form_with_zero_noise():
    diffusion = GaussianDiffusion(timesteps=10, schedule="linear")
    x0 = torch.randn(4, 3, 8, 8)
    t = torch.tensor([0, 1, 5, 9], dtype=torch.long)
    noise = torch.zeros_like(x0)
    xt = diffusion.q_sample(x0, t, noise)
    expected = diffusion.sqrt_alphas_cumprod[t].reshape(4, 1, 1, 1) * x0
    assert torch.allclose(xt, expected)


def test_ddim_and_ddpm_sampling_shapes():
    diffusion = GaussianDiffusion(timesteps=4, schedule="linear")
    model = ZeroModel()
    shape = (2, 3, 8, 8)
    ddim = diffusion.sample_ddim(model, shape, torch.device("cpu"), steps=2)
    ddpm = diffusion.sample_ddpm(model, shape, torch.device("cpu"))
    assert ddim.shape == shape
    assert ddpm.shape == shape
    assert torch.isfinite(ddim).all()
    assert torch.isfinite(ddpm).all()
