# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:43:58 2025

@author: railt
"""

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1. Load metrics
# -------------------------
df = pd.read_csv("slvit_food101_metrics.csv")

epochs     = df["epoch"]
train_loss = df["train_loss"]
test_loss  = df["test_loss"]
train_acc  = df["train_acc"]
test_acc   = df["test_acc"]
lr         = df["lr"]
epoch_time = df["epoch_time"]

# -------------------------
# 2. Loss plot
# -------------------------
plt.figure()
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, test_loss,  label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss (SL-ViT Food101)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("loss_curve.png", dpi=200)

# -------------------------
# 3. Accuracy plot
# -------------------------
plt.figure()
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, test_acc,  label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy (SL-ViT Food101)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("accuracy_curve.png", dpi=200)

# -------------------------
# 4. Learning rate schedule
# -------------------------
plt.figure()
plt.plot(epochs, lr)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Cosine LR Schedule")
plt.grid(True)
plt.tight_layout()
# plt.savefig("lr_schedule.png", dpi=200)

plt.figure()
plt.plot(epochs, epoch_time)
plt.xlabel("Epoch")
plt.ylabel("Seconds per epoch")
plt.title("Epoch Time")
plt.grid(True)
plt.tight_layout()
# plt.savefig("epoch_time.png", dpi=200)

plt.show()