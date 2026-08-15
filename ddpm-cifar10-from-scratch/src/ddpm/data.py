from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def make_cifar10_loader(data_dir: str, batch_size: int, num_workers: int, train: bool = True) -> DataLoader:
    transform_steps = []
    if train:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform = transforms.Compose(transform_steps)
    extracted = Path(data_dir) / "cifar-10-batches-py"
    dataset = datasets.CIFAR10(data_dir, train=train, transform=transform, download=not extracted.exists())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=num_workers > 0,
    )
