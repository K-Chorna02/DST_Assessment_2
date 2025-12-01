# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 23:01:02 2025

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

from Plain18 import Plain18
from torchvision.datasets import CIFAR10
from torchvision import transforms

log = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
    "lr": [],
    "epoch_time": []
}

batch_size = 64

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

train_data = CIFAR10(root='./data', train=True,  download=True, transform=train_tf)
test_data  = CIFAR10(root='./data', train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = Plain18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
base_lr = 1e-3
optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)

epochs = 500
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

best_acc = 0.0
best_acc_state = None

best_loss = float("inf")
best_loss_state = None

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

    if test_acc > best_acc:
        best_acc = test_acc
        best_acc_state = model.state_dict()
        torch.save(best_acc_state, "plain18_cifar10_best_acc.pth")

    if test_loss < best_loss:
        best_loss = test_loss
        best_loss_state = model.state_dict()
        torch.save(best_loss_state, "plain18_cifar10_best_loss.pth")

    print(
        f"[Epoch {epoch+1}/{epochs}] "
        f"LR={lr_now:.6f} "
        f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
        f"Test  Loss={test_loss:.4f} Acc={test_acc:.4f} | "
        f"Best Acc={best_acc:.4f} Best Loss={best_loss:.4f}"
    )

df = pd.DataFrame(log)
df.to_csv("plain18_cifar10_metrics.csv", index=False)

torch.save(model.state_dict(), "plain18_cifar10_final.pth")

print("Best model (by test acc)  → plain18_cifar10_best_acc.pth with acc  =", best_acc)
print("Best model (by test loss) → plain18_cifar10_best_loss.pth with loss =", best_loss)
