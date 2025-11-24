# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 11:22:28 2025

@author: railt
"""

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load CSV logs
# -------------------------
resnet_path = "resnet18_cifar10_metrics.csv"
vit_path    = "slvit_cifar10_metrics.csv"

resnet = pd.read_csv(resnet_path)
vit    = pd.read_csv(vit_path)

# Optional: print head to double-check
print("ResNet18 metrics:")
print(resnet.head())
print("\nViT metrics:")
print(vit.head())

# -------------------------
# Helper function for plotting one metric
# -------------------------
def plot_metric(ax, resnet_df, vit_df, metric, ylabel, title):
    ax.plot(resnet_df["epoch"], resnet_df[metric], label="ResNet18")
    ax.plot(vit_df["epoch"], vit_df[metric], label="ViT")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

# -------------------------
# 1) Loss & accuracy in a 2x2 grid
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

plot_metric(
    axes[0, 0],
    resnet, vit,
    metric="train_loss",
    ylabel="Train Loss",
    title="Train Loss vs Epoch"
)

plot_metric(
    axes[0, 1],
    resnet, vit,
    metric="test_loss",
    ylabel="Test Loss",
    title="Test Loss vs Epoch"
)

plot_metric(
    axes[1, 0],
    resnet, vit,
    metric="train_acc",
    ylabel="Train Accuracy",
    title="Train Accuracy vs Epoch"
)

plot_metric(
    axes[1, 1],
    resnet, vit,
    metric="test_acc",
    ylabel="Test Accuracy",
    title="Test Accuracy vs Epoch"
)

plt.tight_layout()
plt.show()

# -------------------------
# 2) Learning rate & epoch time
# -------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

plot_metric(
    axes2[0],
    resnet, vit,
    metric="lr",
    ylabel="Learning Rate",
    title="Learning Rate vs Epoch"
)

plot_metric(
    axes2[1],
    resnet, vit,
    metric="epoch_time",
    ylabel="Seconds",
    title="Epoch Time vs Epoch"
)

plt.tight_layout()
plt.show()

# -------------------------
# (Optional) Save figures
# -------------------------
# fig.savefig("cifar10_resnet_vs_vit_loss_acc.png", dpi=300)
# fig2.savefig("cifar10_resnet_vs_vit_lr_time.png", dpi=300)
