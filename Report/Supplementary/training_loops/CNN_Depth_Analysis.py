import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from tqdm import tqdm
import pandas as pd



# Using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Dataset (CIFAR-10)
batch_size = 128

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
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

train_data = CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_data  = CIFAR10(root="./data", train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


# Simple CNN (1, 3, 5 Conv layers)
class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers):
        super().__init__()
        assert num_conv_layers in [1, 3, 5]

        ch_list = [32, 64, 128, 256, 256]
        conv_blocks = []
        in_ch = 3

        for i in range(num_conv_layers):
            out_ch = ch_list[i]
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            )
            in_ch = out_ch

        self.features = nn.Sequential(*conv_blocks)

        spatial = 32 // (2 ** num_conv_layers)
        self.classifier = nn.Linear(in_ch * spatial * spatial, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Training utilities
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


def eval_model(model, loader, loss_fn):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(False)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    torch.set_grad_enabled(True)
    return running_loss / total, correct / total


# Experiment runner
def run_experiment(num_conv_layers, num_epochs=15, lr=1e-3, weight_decay=1e-4):
    print(f"\n=== {num_conv_layers} CONV LAYERS ===")

    model = SimpleCNN(num_conv_layers).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    log = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
        "epoch_time": []
    }

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        test_loss, test_acc = eval_model(model, test_loader, loss_fn)
        epoch_time = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]

        # Save logs
        log["epoch"].append(epoch)
        log["train_loss"].append(train_loss)
        log["train_acc"].append(train_acc)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)
        log["lr"].append(lr_current)
        log["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={train_acc*100:.2f}% | test_acc={test_acc*100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc

    # Save CSV
    df = pd.DataFrame(log)
    df.to_csv(f"results_cnn_{num_conv_layers}layers.csv", index=False)
    print(f"Saved metrics → results_cnn_{num_conv_layers}layers.csv")
    print(f"Best accuracy = {best_acc*100:.2f}%")

    return best_acc

# Run all experiments
if __name__ == "__main__":
    results = {}

    for layers in [1, 3, 5]:
        acc = run_experiment(num_conv_layers=layers, num_epochs=15)
        results[layers] = acc

    print("\n=== Summary ===")
    for layers, acc in results.items():
        print(f"{layers} conv layers → {acc*100:.2f}%")