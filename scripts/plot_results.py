"""Gera gráficos comparativos de desempenho dos experimentos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Gera gráficos de comparação")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    history = pd.read_csv(args.results_dir / "training_history.csv")
    with (args.results_dir / "summary.json").open("r", encoding="utf-8") as fp:
        summary = pd.DataFrame(json.load(fp))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="dataset", y="test_accuracy", hue="model")
    plt.title("Acurácia de teste por dataset e modelo")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.output_dir / "test_accuracy_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=history, x="epoch", y="val_accuracy", hue="model", style="dataset", markers=True)
    plt.title("Evolução da acurácia de validação")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.output_dir / "val_accuracy_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=history, x="epoch", y="train_loss", hue="model", style="dataset", markers=True)
    plt.title("Evolução da perda de treino")
    plt.tight_layout()
    plt.savefig(args.output_dir / "train_loss_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=history, x="epoch", y="val_loss", hue="model", style="dataset", markers=True)
    plt.title("Evolução da perda de validação")
    plt.tight_layout()
    plt.savefig(args.output_dir / "val_loss_curves.png", dpi=200)
    plt.close()

    print(f"Gráficos gerados em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
