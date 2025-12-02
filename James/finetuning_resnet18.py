# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 09:22:06 2025

@author: railt
"""

import time
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

num_classes = 10
batch_size = 64
epochs = 30
base_lr = 1e-3
weight_decay = 1e-2

weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    preprocess,
])

test_transform = preprocess

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader64 = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
test_loader64 = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)

resnet18_ft = resnet18(weights=weights)

for name, param in resnet18_ft.named_parameters():
    if "layer1" in name or "layer2" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

resnet18_ft.fc = nn.Linear(resnet18_ft.fc.in_features, num_classes)
resnet18_ft = resnet18_ft.to(device)

criterion = nn.CrossEntropyLoss()

trainable_params = filter(lambda p: p.requires_grad, resnet18_ft.parameters())
optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)

scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

Train_loss_res18_ft = []
Train_acc_res18_ft = []
Test_loss_res18_ft = []
Test_acc_res18_ft = []
Epoch_time_res18_ft = []

best_acc = 0.0
best_state = None

total_start = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    resnet18_ft.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader64,
                               desc=f"Epoch {epoch+1}/{epochs}",
                               dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_train_loss = running_loss / total
    epoch_train_acc = running_correct / total
    Train_loss_res18_ft.append(epoch_train_loss)
    Train_acc_res18_ft.append(epoch_train_acc)

    resnet18_ft.eval()
    test_loss, correct, total_test = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader64:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet18_ft(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total_test += labels.size(0)

    epoch_test_loss = test_loss / total_test
    epoch_test_acc = correct / total_test
    Test_loss_res18_ft.append(epoch_test_loss)
    Test_acc_res18_ft.append(epoch_test_acc)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    epoch_end = time.time()
    Epoch_time_res18_ft.append(epoch_end - epoch_start)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"LR={current_lr:.6f} | "
        f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc*100:.2f}% | "
        f"Test Loss={epoch_test_loss:.4f}, Test Acc={epoch_test_acc*100:.2f}% | "
        f"Time={epoch_end - epoch_start:.2f}s"
    )

    if epoch_test_acc > best_acc:
        best_acc = epoch_test_acc
        best_state = resnet18_ft.state_dict().copy()

total_end = time.time()
print(f"Total training time: {total_end - total_start:.2f}s")

results = {
    "Train_loss": Train_loss_res18_ft,
    "Train_acc": Train_acc_res18_ft,
    "Test_loss": Test_loss_res18_ft,
    "Test_acc": Test_acc_res18_ft,
    "Epoch_time": Epoch_time_res18_ft
}

with open("resnet18_finetune_cifar10_metrics.pkl", "wb") as f:
    pickle.dump(results, f)

torch.save(resnet18_ft.state_dict(), "resnet18_finetune_cifar10_final.pth")
print("Saved final model → resnet18_finetune_cifar10_final.pth")

if best_state is not None:
    torch.save(best_state, "resnet18_finetune_cifar10_bestacc.pth")
    print(f"Saved best-acc model → resnet18_finetune_cifar10_bestacc.pth "
          f"(acc = {best_acc*100:.2f}%)")
