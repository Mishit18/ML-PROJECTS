from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def make_cifar10_loader(data_dir: str, batch_size: int, num_workers: int, train: bool = True) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(data_dir, train=train, transform=transform, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=num_workers > 0,
    )
