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
from torchvision.models import resnet18, ResNet18_Weights

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
# Data (CIFAR-10 for ImageNet ResNet18)
# -------------------------
batch_size = 64

# Pretrained ImageNet weights + official preprocessing (Resize/CenterCrop/Normalize)
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

# Training = flips + preprocess
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    preprocess,
])

# Test = only preprocess
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
# Model (ResNet18 transfer learning)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Load pretrained ResNet18
model = resnet18(weights=weights)

# Replace classifier head for CIFAR-10
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model = model.to(device)

# --------- Freeze backbone? ----------
freeze_backbone = True  # set False if you want full fine-tuning

if freeze_backbone:
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
    print("Backbone frozen. Training FC layer only.")
else:
    print("Fine-tuning entire ResNet18.")

# -------------------------
# Optimizer & Scheduler
# -------------------------
criterion = nn.CrossEntropyLoss()
base_lr = 1e-3

trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=1e-2)

epochs = 30
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# -------------------------
# Best trackers
# -------------------------
best_acc = 0.0
best_loss = float("inf")

best_acc_path  = "resnet18_cifar10_transfer_best_acc.pth"
best_loss_path = "resnet18_cifar10_transfer_best_loss.pth"
final_path     = "resnet18_cifar10_transfer_final.pth"
metrics_csv    = "resnet18_cifar10_transfer_metrics.csv"

# -------------------------
# Training Loop
# -------------------------
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

        train_pbar.set_postfix({"image": batch_idx * batch_size})

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

            val_pbar.set_postfix({"image": batch_idx * batch_size})

    test_loss = val_loss_total / len(test_data)
    test_acc = val_correct / len(test_data)

    # ---- Scheduler ----
    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]
    epoch_time = time.time() - start_time

    # ---- Logging ----
    log["epoch"].append(epoch + 1)
    log["train_loss"].append(train_loss)
    log["train_acc"].append(train_acc)
    log["test_loss"].append(test_loss)
    log["test_acc"].append(test_acc)
    log["lr"].append(lr_now)
    log["epoch_time"].append(epoch_time)

    # ---- Save best accuracy model ----
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_acc_path)
        print(f"Saved BEST ACC model ({best_acc:.4f}) → {best_acc_path}")

    # ---- Save best loss model ----
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), best_loss_path)
        print(f"Saved BEST LOSS model ({best_loss:.4f}) → {best_loss_path}")

    print(
        f"[Epoch {epoch + 1}/{epochs}] "
        f"LR={lr_now:.6f} | "
        f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
        f"Test  Loss={test_loss:.4f} Acc={test_acc:.4f} | "
        f"Best Acc={best_acc:.4f} Best Loss={best_loss:.4f}"
    )

# -------------------------
# Save final model & metrics
# -------------------------
torch.save(model.state_dict(), final_path)
print(f"Saved final model → {final_path}")

pd.DataFrame(log).to_csv(metrics_csv, index=False)
print(f"Saved metrics → {metrics_csv}")

print(f"\nBest ACC model:  {best_acc_path}")
print(f"Best LOSS model: {best_loss_path}")
print(f"Final model:     {final_path}")

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
from torchvision.models import resnet18, ResNet18_Weights

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
# Data (CIFAR-10 for ImageNet ResNet18)
# -------------------------
batch_size = 64

# Load pretrained weights + associated transforms
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()  # includes Resize/CenterCrop/ToTensor/Normalize

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
# Model (ResNet18 transfer learning)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pretrained ImageNet ResNet18
model = resnet18(weights=weights)

# Replace final layer for CIFAR-10 (10 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

# --------- Toggle: freeze backbone or fine-tune all ----------
freeze_backbone = True  # set to False if you want to fine-tune entire model

if freeze_backbone:
    for name, param in model.named_parameters():
        # freeze all except final classifier
        if not name.startswith("fc."):
            param.requires_grad = False
    print("Backbone frozen, training only final fc layer.")
else:
    print("Fine-tuning entire ResNet18.")

# -------------------------
# Optimizer & Scheduler
# -------------------------
criterion = nn.CrossEntropyLoss()
base_lr = 1e-3

# Only optimize parameters that require grad (handles freeze_backbone=True)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=1e-2)

epochs = 100  # You usually don't need 500 epochs for transfer learning
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
        torch.save(best_acc_state, "resnet18_cifar10_transfer_best_acc.pth")

    # ---- SAVE BEST BY VAL LOSS ----
    if test_loss < best_loss:
        best_loss = test_loss
        best_loss_state = model.state_dict()
        torch.save(best_loss_state, "resnet18_cifar10_transfer_best_loss.pth")

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
df.to_csv("resnet18_cifar10_transfer_metrics.csv", index=False)
print("Saved metrics → resnet18_cifar10_transfer_metrics.csv")

torch.save(model.state_dict(), "resnet18_cifar10_transfer_final.pth")
print("Saved final model → resnet18_cifar10_transfer_final.pth")

print(
    "Best model (by test acc) → resnet18_cifar10_transfer_best_acc.pth "
    f"with acc = {best_acc}"
)
print(
    "Best model (by test loss) → resnet18_cifar10_transfer_best_loss.pth "
    f"with loss = {best_loss}"
)
