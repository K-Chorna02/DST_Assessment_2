import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pickle

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
num_classes = 10
batch_size = 128
epochs = 100
learning_rate = 1e-4

# Data transforms for CIFAR10
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# CIFAR10 dataset
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader64 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader64 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load pretrained ResNet18
resnet18_ft = models.resnet18(pretrained=True)

# Freeze early layers
for name, param in resnet18_ft.named_parameters():
    if "layer1" in name or "layer2" in name:
        param.requires_grad = False

# Replace classifier
resnet18_ft.fc = nn.Linear(resnet18_ft.fc.in_features, num_classes)
resnet18_ft = resnet18_ft.to(device)

# Optimizer and loss
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet18_ft.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Metrics lists
Train_loss_res18_ft = []
Train_acc_res18_ft = []
Test_loss_res18_ft = []
Test_acc_res18_ft = []
Epoch_time_res18_ft = []

# Track best accuracy
best_acc = 0.0
best_state = None

# Training
total_start = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    resnet18_ft.train()
    running_loss, running_correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader64, desc=f"Epoch {epoch+1}/{epochs}"):
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

    # Testing
    resnet18_ft.eval()
    test_loss, correct, total_test = 0, 0, 0

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

    # Save best model
    if epoch_test_acc > best_acc:
        best_acc = epoch_test_acc
        best_state = resnet18_ft.state_dict().copy()

    # Time
    epoch_end = time.time()
    Epoch_time_res18_ft.append(epoch_end - epoch_start)

    print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc*100:.2f}% | "
          f"Test Loss={epoch_test_loss:.4f}, Test Acc={epoch_test_acc*100:.2f}% | Time={epoch_end - epoch_start:.2f}s")

total_end = time.time()
print(f"Total training time: {total_end - total_start:.2f}s")

# Save metrics dictionary
results = {
    "Train_loss": Train_loss_res18_ft,
    "Train_acc": Train_acc_res18_ft,
    "Test_loss": Test_loss_res18_ft,
    "Test_acc": Test_acc_res18_ft,
    "Epoch_time": Epoch_time_res18_ft
}

with open("resnet18_finetune_cifar10_metrics.pkl", "wb") as f:
    pickle.dump(results, f)

# Save final weights
torch.save(resnet18_ft.state_dict(), "resnet18_finetune_cifar10_final.pth")

# Save best accuracy weights
if best_state is not None:
    torch.save(best_state, "resnet18_finetune_cifar10_bestacc.pth")
