"""Treino/avaliação da CNN customizada e ResNet50 em MNIST/CIFAR10."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from src.cnn_model import CustomCNN, OptimizationConfig


DATASETS = ("MNIST", "CIFAR10")
MODELS = ("CustomCNN", "ResNet50")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(dataset_name: str, for_resnet: bool, train: bool) -> transforms.Compose:
    """Monta pipeline de transformações para dataset, modo treino e arquitetura."""
    if dataset_name == "MNIST":
        if for_resnet:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            aug = [transforms.RandomRotation(8)] if train else []
            return transforms.Compose(
                aug
                + [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        aug = [transforms.RandomRotation(8)] if train else []
        return transforms.Compose(aug + [transforms.ToTensor(), normalize])

    if for_resnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] if train else []
        return transforms.Compose(aug + [transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    aug = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] if train else []
    return transforms.Compose(aug + [transforms.ToTensor(), normalize])


def load_dataset(name: str, transform: transforms.Compose, train: bool) -> datasets.VisionDataset:
    dataset_cls = datasets.MNIST if name == "MNIST" else datasets.CIFAR10
    return dataset_cls(root="data", train=train, download=True, transform=transform)


def create_loaders(
    dataset_name: str,
    for_resnet: bool,
    batch_size: int,
    seed: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = load_dataset(dataset_name, build_transforms(dataset_name, for_resnet, train=True), train=True)
    test_dataset = load_dataset(dataset_name, build_transforms(dataset_name, for_resnet, train=False), train=False)

    val_size = max(1, int(val_split * len(train_dataset)))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


def make_model(model_name: str, dataset_name: str, device: torch.device, dropout_rate: float) -> nn.Module:
    if model_name == "CustomCNN":
        in_channels = 1 if dataset_name == "MNIST" else 3
        model = CustomCNN(in_channels=in_channels, num_classes=10, dropout_rate=dropout_rate)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    model.train(mode=train)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            if optimizer is None:
                raise ValueError("optimizer is required when train=True.")
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_and_evaluate(
    dataset_name: str,
    model_name: str,
    device: torch.device,
    config: OptimizationConfig,
    seed: int,
) -> Tuple[Dict, list]:
    train_loader, val_loader, test_loader = create_loaders(
        dataset_name=dataset_name,
        for_resnet=(model_name == "ResNet50"),
        batch_size=config.batch_size,
        seed=seed,
        val_split=config.val_split,
        num_workers=config.num_workers,
    )

    model = make_model(model_name, dataset_name, device, config.dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    epoch_rows = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        epoch_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
    return summary, epoch_rows


def save_results(output_dir: Path, summaries: list, epochs: list):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summaries, fp, indent=2)

    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["dataset", "model", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(epochs)


def parse_args():
    parser = argparse.ArgumentParser(description="Executa experimentos CNN x ResNet50")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.val_split < 1:
        raise ValueError("--val-split must be between 0 and 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OptimizationConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    summaries, all_epochs = [], []
    for dataset_name in DATASETS:
        for model_name in MODELS:
            summary, epochs = train_and_evaluate(dataset_name, model_name, device, config, args.seed)
            summaries.append(summary)
            all_epochs.extend(epochs)
            print(
                f"[{dataset_name}][{model_name}] "
                f"test_acc={summary['test_accuracy']:.4f} test_loss={summary['test_loss']:.4f}"
            )

    save_results(args.output_dir, summaries, all_epochs)
    print(f"Resultados salvos em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
