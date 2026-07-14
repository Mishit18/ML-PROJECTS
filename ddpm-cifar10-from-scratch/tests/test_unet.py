import torch

from src.ddpm import UNet, count_parameters


def test_unet_output_shape_matches_input():
    model = UNet(
        in_channels=3,
        base_channels=16,
        channel_mults=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[16],
        num_heads=4,
        image_size=32,
    )
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0, 3], dtype=torch.long)
    y = model(x, t)
    assert y.shape == x.shape
    assert count_parameters(model) > 0
