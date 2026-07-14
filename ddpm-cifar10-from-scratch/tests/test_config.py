from pathlib import Path

from src.ddpm.config import validate_config
from src.ddpm.utils import load_config


def test_all_configs_validate():
    for path in Path("configs").glob("*.yaml"):
        validate_config(load_config(path))
