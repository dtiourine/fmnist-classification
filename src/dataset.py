from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

app = typer.Typer()


def get_data(data_dir=RAW_DATA_DIR, batch_size=64):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 0.5)
        ),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset_augmented = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    full_train_dataset_clean = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=val_test_transform
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=val_test_transform
    )

    train_size = int(0.8 * len(full_train_dataset_augmented))
    val_size = len(full_train_dataset_augmented) - train_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_train_dataset_augmented)),
        [train_size, val_size],
        generator=generator
    )

    train_dataset = torch.utils.data.Subset(full_train_dataset_augmented, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset_clean, val_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


@app.command()
def main():
    get_data()


if __name__ == "__main__":
    app()
