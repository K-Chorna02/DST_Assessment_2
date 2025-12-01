# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 22:23:41 2025

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

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from SL_Vit import ViT
from torchvision.datasets import Food101
from torchvision import transforms

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
    "epoch_time": []
}

# -------------------------
# Data
# -------------------------
image_size = 64
batch_size = 64

train_tf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

test_tf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

train_data = Food101(root='./data', split='train', download=True, transform=train_tf)
test_data  = Food101(root='./data', split='test',  download=True, transform=test_tf)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

# -------------------------
# Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ViT(
    image_size = 64,
    patch_size = 8,          # 64 / 8 = 8, so 8×8 = 64 patches
    num_classes = 101,       # Food101
    dim = 256,               # embedding dim
    depth = 6,               # number of Transformer blocks
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

# -------------------------
# Optimizer & Scheduler
# -------------------------
criterion = nn.CrossEntropyLoss()
base_lr = 1e-4
optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)

epochs = 100

# Cosine LR schedule over 100 epochs
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

best_acc = 0.0
best_state = None

for epoch in range(epochs):
    start_time = time.time()

    model.train()
    total_loss, total_correct = 0.0, 0

    train_pbar = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f"Train Epoch {epoch+1}/{epochs}",
        dynamic_ncols=True
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

        # current image index (robust even if last batch smaller)
        current_image = (batch_idx - 1) * batch_size + imgs.size(0)
        train_pbar.set_postfix({"image": current_image})

    train_loss = total_loss / len(train_data)
    train_acc  = total_correct / len(train_data)

    model.eval()
    val_loss_total, val_correct = 0.0, 0

    val_pbar = tqdm(
        enumerate(test_loader, start=1),
        total=len(test_loader),
        desc=f"Val   Epoch {epoch+1}/{epochs}",
        dynamic_ncols=True
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
    test_acc  = val_correct / len(test_data)

    # Step the LR scheduler AFTER validation
    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]

    epoch_time = time.time() - start_time


    log["epoch"].append(epoch + 1)
    log["train_loss"].append(train_loss)
    log["train_acc"].append(train_acc)
    log["test_loss"].append(test_loss)
    log["test_acc"].append(test_acc)
    log["lr"].append(lr_now)
    log["epoch_time"].append(epoch_time)

    # ----------------------
    # Track best model
    # ----------------------
    if test_acc > best_acc:
        best_acc = test_acc
        best_state = model.state_dict()
        torch.save(best_state, "slvit_food101_best.pth")

    print(
        f"[Epoch {epoch+1}/{epochs}] "
        f"LR={lr_now:.6f} "
        f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
        f"Test Loss={test_loss:.4f} Acc={test_acc:.4f} | "
        f"Best Acc={best_acc:.4f}"
    )

# -------------------------
# Save metrics and final model
# -------------------------
df = pd.DataFrame(log)
df.to_csv("slvit_food101_metrics.csv", index=False)
print("Saved metrics → slvit_food101_metrics.csv")

torch.save(model.state_dict(), "slvit_food101_final.pth")
print("Saved final model → slvit_food101_final.pth")
print("Best model (by test acc) → slvit_food101_best.pth with acc =", best_acc)
