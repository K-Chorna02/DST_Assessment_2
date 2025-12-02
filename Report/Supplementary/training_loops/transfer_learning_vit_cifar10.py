# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 23:36:45 2025

@author: railt
"""

from math import sqrt
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# -------------------------
# Logging dict for metrics
# -------------------------
log = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
    "lr": [],
    "epoch_time": [],
}

# -------------------------
# Data (CIFAR-10 for ViT-B/16)
# -------------------------
batch_size = 64

# Load pretrained weights + associated transforms
weights = ViT_B_16_Weights.IMAGENET1K_V1
preprocess = weights.transforms()  # usually Resize(224), CenterCrop(224), ToTensor, Normalize

# For training, we add augmentation *before* the official preprocess
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    preprocess,  # includes Resize + CenterCrop + ToTensor + Normalize
])

# For test, just use the official preprocess
test_tf = preprocess

train_data = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_tf,
)
test_data = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_tf,
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# -------------------------
# Model (ViT-B/16 transfer learning)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Load pretrained ImageNet ViT-B/16
model = vit_b_16(weights=weights)

# Replace head for CIFAR-10 (10 classes)
in_features = model.heads[-1].in_features  # typically 768
model.heads = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 10),
)
model = model.to(device)

# --------- Toggle: freeze backbone or fine-tune all ----------
freeze_backbone = True  # set to False if you want to fine-tune entire ViT

if freeze_backbone:
    for name, param in model.named_parameters():
        # freeze all except classifier head
        if not name.startswith("heads."):
            param.requires_grad = False
    print("Backbone frozen, training only ViT heads.")
else:
    print("Fine-tuning entire ViT-B/16.")

# -------------------------
# Optimizer & Scheduler
# -------------------------
criterion = nn.CrossEntropyLoss()
base_lr = 1e-3

# Only optimize parameters that require grad (handles freeze_backbone=True)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=1e-2)

epochs = 30
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# -------------------------
# Best trackers (acc & loss)
# -------------------------
best_acc = 0.0
best_acc_state = None
best_loss = float("inf")
best_loss_state = None

for epoch in range(epochs):
    start_time = time.time()

    # ---- TRAIN ----
    model.train()
    total_loss, total_correct = 0.0, 0

    train_pbar = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f"Train Epoch {epoch + 1}/{epochs}",
        dynamic_ncols=True,
    )

    for batch_idx, (imgs, labels) in train_pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds.argmax(1) == labels).sum().item()

        current_image = (batch_idx - 1) * batch_size + imgs.size(0)
        train_pbar.set_postfix({"image": current_image})

    train_loss = total_loss / len(train_data)
    train_acc = total_correct / len(train_data)

    # ---- EVAL ----
    model.eval()
    val_loss_total, val_correct = 0.0, 0

    val_pbar = tqdm(
        enumerate(test_loader, start=1),
        total=len(test_loader),
        desc=f"Val Epoch {epoch + 1}/{epochs}",
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch_idx, (imgs, labels) in val_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            val_loss_total += loss.item() * imgs.size(0)
            val_correct += (preds.argmax(1) == labels).sum().item()

            current_image = (batch_idx - 1) * batch_size + imgs.size(0)
            val_pbar.set_postfix({"image": current_image})

    test_loss = val_loss_total / len(test_data)
    test_acc = val_correct / len(test_data)

    # ---- SCHEDULER STEP ----
    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]
    epoch_time = time.time() - start_time

    # ---- LOGGING ----
    log["epoch"].append(epoch + 1)
    log["train_loss"].append(train_loss)
    log["train_acc"].append(train_acc)
    log["test_loss"].append(test_loss)
    log["test_acc"].append(test_acc)
    log["lr"].append(lr_now)
    log["epoch_time"].append(epoch_time)

    # ---- SAVE BEST BY ACC ----
    if test_acc > best_acc:
        best_acc = test_acc
        best_acc_state = model.state_dict()
        torch.save(best_acc_state, "vit_b16_cifar10_transfer_best_acc.pth")

    # ---- SAVE BEST BY VAL LOSS ----
    if test_loss < best_loss:
        best_loss = test_loss
        best_loss_state = model.state_dict()
        torch.save(best_loss_state, "vit_b16_cifar10_transfer_best_loss.pth")

    print(
        f"[Epoch {epoch + 1}/{epochs}] "
        f"LR={lr_now:.6f} "
        f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
        f"Test Loss={test_loss:.4f} Acc={test_acc:.4f} | "
        f"Best Acc={best_acc:.4f} Best Loss={best_loss:.4f}"
    )

# -------------------------
# Save metrics and final model
# -------------------------
df = pd.DataFrame(log)
df.to_csv("vit_b16_cifar10_transfer_metrics.csv", index=False)
print("Saved metrics → vit_b16_cifar10_transfer_metrics.csv")

torch.save(model.state_dict(), "vit_b16_cifar10_transfer_final.pth")
print("Saved final model → vit_b16_cifar10_transfer_final.pth")

print(
    "Best model (by test acc) → vit_b16_cifar10_transfer_best_acc.pth "
    f"with acc = {best_acc}"
)
print(
    "Best model (by test loss) → vit_b16_cifar10_transfer_best_loss.pth "
    f"with loss = {best_loss}"
)
