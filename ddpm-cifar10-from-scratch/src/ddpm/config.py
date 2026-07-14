from __future__ import annotations


def validate_config(config: dict) -> None:
    required_top = ["seed", "data_dir", "out_dir", "image_size", "channels", "batch_size", "model", "diffusion", "train"]
    missing = [key for key in required_top if key not in config]
    if missing:
        raise ValueError(f"Missing top-level config keys: {missing}")

    model = config["model"]
    train = config["train"]
    diffusion = config["diffusion"]

    if config["image_size"] != 32:
        raise ValueError("This project is tuned for CIFAR-10 at image_size=32")
    if config["channels"] != 3:
        raise ValueError("CIFAR-10 configs should use channels=3")
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if train.get("grad_accum_steps", 1) <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if train["lr"] <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 < train["ema_decay"] < 1:
        raise ValueError("ema_decay must be in (0, 1)")
    if diffusion["timesteps"] <= 0:
        raise ValueError("diffusion.timesteps must be positive")
    if diffusion["schedule"] not in {"linear", "cosine"}:
        raise ValueError("diffusion.schedule must be 'linear' or 'cosine'")
    if model["base_channels"] % model["num_heads"] != 0:
        raise ValueError("base_channels must be divisible by num_heads")
    for mult in model["channel_mults"]:
        channels = model["base_channels"] * mult
        if channels % model["num_heads"] != 0:
            raise ValueError(f"attention channels {channels} must be divisible by num_heads")
    for resolution in model["attention_resolutions"]:
        if resolution not in {4, 8, 16, 32}:
            raise ValueError(f"unexpected attention resolution: {resolution}")
