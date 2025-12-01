import time, os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import Food101
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

log = {
    "epoch": [], "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []
}

food_root = "./data"
batch_size = 128

train_tf = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = Food101(root=food_root, split="train", download=False, transform=train_tf)
val_dataset   = Food101(root=food_root, split="test",  download=False, transform=val_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Classes:", len(train_dataset.classes))

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, 101))

for p in model.parameters():
    p.requires_grad = False

for name, p in model.named_parameters():
    if any(x in name for x in ["layer4", "fc"]):
        p.requires_grad = True

model = model.to(device)

lr = 1e-3
weight_decay = 1e-3
num_epochs = 30

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=5, gamma=0.2)

best_val_acc = 0.0
best_model_path = "food101_resnet18_best.pth"
final_model_path = "food101_resnet18_final.pth"

start_time = time.time()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()

    model.train()
    train_loss_epoch = 0
    train_correct = 0
    train_total = 0

    for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item() * X.size(0)
        train_correct += (logits.argmax(dim=1) == y).sum().item()
        train_total += y.size(0)

    train_loss = train_loss_epoch / train_total
    train_acc = 100 * train_correct / train_total

    model.eval()
    val_loss_epoch = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X, y in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]", leave=False):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            val_loss_epoch += loss.item() * X.size(0)
            val_correct += (logits.argmax(dim=1) == y).sum().item()
            val_total += y.size(0)

    val_loss = val_loss_epoch / val_total
    val_acc = 100 * val_correct / val_total

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = (time.time() - epoch_start) / 60
    total_time = (time.time() - start_time) / 60

    log["epoch"].append(epoch)
    log["train_loss"].append(train_loss)
    log["train_acc"].append(train_acc)
    log["val_loss"].append(val_loss)
    log["val_acc"].append(val_acc)
    log["lr"].append(current_lr)
    log["epoch_time"].append(epoch_time)

    print(f"Epoch {epoch}/{num_epochs} | time {epoch_time:.2f}m | lr {current_lr:.5f} | "
          f"train_loss {train_loss:.4f} acc {train_acc:.2f}% | val_loss {val_loss:.4f} acc {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model ({best_val_acc:.2f}%)")

out_dir = "."
os.makedirs(out_dir, exist_ok=True)

log_df = pd.DataFrame(log)
log_df.to_csv("food101_resnet18_log.csv", index=False)

torch.save(model.state_dict(), final_model_path)

print("Training complete.")
print("Best model saved to:", best_model_path)
print("Final model saved to:", final_model_path)


