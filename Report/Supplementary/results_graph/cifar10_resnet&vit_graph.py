# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 14:12:07 2025

@author: railt
"""

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load metrics
# -------------------------
resnet_df = pd.read_csv("resnet18_cifar10_metrics.csv")
vit_df    = pd.read_csv("slvit_cifar10_metrics.csv")

# Make sure epoch is sorted (just in case)
resnet_df = resnet_df.sort_values("epoch")
vit_df    = vit_df.sort_values("epoch")

# -------------------------
# Helper plotting function
# -------------------------
def plot_metric(
    resnet_df,
    vit_df,
    metric_train,
    metric_test,
    ylabel,
    title,
    filename=None
):
    plt.figure(figsize=(8, 5))

    # ResNet18
    plt.plot(
        resnet_df["epoch"],
        resnet_df[metric_train],
        label=f"ResNet18 {metric_train} (train)",

    )
    plt.plot(
        resnet_df["epoch"],
        resnet_df[metric_test],
        label=f"ResNet18 {metric_test} (test)",

    )

    # ViT
    plt.plot(
        vit_df["epoch"],
        vit_df[metric_train],
        label=f"ViT {metric_train} (train)",

    )
    plt.plot(
        vit_df["epoch"],
        vit_df[metric_test],
        label=f"ViT {metric_test} (test)",

    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()

# -------------------------
# 1) Loss vs epoch (train + test)
# -------------------------
plot_metric(
    resnet_df,
    vit_df,
    metric_train="train_loss",
    metric_test="test_loss",
    ylabel="Loss",
    title="CIFAR-10: Loss vs Epoch (ResNet18 vs ViT)",
    filename="cifar10_resnet_vs_vit_loss.png"
)

# -------------------------
# 2) Accuracy vs epoch (train + test)
# -------------------------
plot_metric(
    resnet_df,
    vit_df,
    metric_train="train_acc",
    metric_test="test_acc",
    ylabel="Accuracy",
    title="CIFAR-10: Accuracy vs Epoch (ResNet18 vs ViT)",
    filename="cifar10_resnet_vs_vit_acc.png"
)

# -------------------------
# 3) (Optional) Learning rate vs epoch
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(resnet_df["epoch"], resnet_df["lr"], label="ResNet18 LR")
plt.plot(vit_df["epoch"], vit_df["lr"], label="ViT LR", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.title("CIFAR-10: Learning Rate Schedule (Cosine Annealing)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar10_resnet_vs_vit_lr.png", dpi=300)
plt.show()
