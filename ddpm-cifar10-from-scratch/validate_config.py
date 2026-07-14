import argparse
import glob
from pathlib import Path

from src.ddpm.config import validate_config
from src.ddpm.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    paths = []
    for pattern in args.configs:
        matches = glob.glob(pattern)
        paths.extend(matches or [pattern])

    for path in paths:
        config = load_config(path)
        validate_config(config)
        print(f"ok: {Path(path)}")


if __name__ == "__main__":
    main()
