import sys
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import entropy

import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF

from tqdm.auto import tqdm

# Add your local model folder to path and import architectures
sys.path.append(r"C:\Users\railt\DST_Assessment_2\James\Model")
from Resnet18 import ResNet18
from SL_Vit import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEVERITY_LEVELS = [1, 2, 3, 4, 5]

# =========================
# Corruption functions
# =========================

def add_noise(images, severity):
    noise_level = 0.05 * severity
    noise = torch.randn_like(images) * noise_level
    return torch.clamp(images + noise, -1, 1).float()

def change_brightness(images, severity):
    factor = 1 - 0.15 * severity
    return torch.clamp(images * factor, -1, 1).float()

def defocus_blur(images, severity):
    kernel_size = 2 * severity + 1  # 3, 5, 7, 9, 11
    blurred = []
    for img in images:
        npimg = img.permute(1, 2, 0).cpu().numpy()
        npimg_blur = cv2.GaussianBlur(npimg, (kernel_size, kernel_size), 0)
        blurred.append(torch.tensor(npimg_blur).permute(2, 0, 1))
    return torch.stack(blurred).clamp(-1, 1).float()

def pixelate(images, severity):
    scale_down = 2 + severity * 2    # 4, 6, 8, 10, 12
    pixelated = []
    for img in images:
        npimg = img.permute(1, 2, 0).cpu().numpy()
        h, w = npimg.shape[:2]
        small = cv2.resize(
            npimg,
            (w // scale_down, h // scale_down),
            interpolation=cv2.INTER_LINEAR
        )
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        pixelated.append(torch.tensor(big).permute(2, 0, 1))
    return torch.stack(pixelated).clamp(-1, 1).float()

def add_fog(images, severity):
    fog_strength = 0.1 * severity
    foggy = []
    for img in images:
        npimg = img.cpu().numpy()
        _, h, w = npimg.shape
        fog = np.random.normal(0.5, 0.2, (3, h, w))
        foggy.append(torch.tensor(npimg + fog_strength * fog))
    return torch.stack(foggy).clamp(-1, 1).float()

def frost(images, severity):
    frost_strength = 0.15 * severity
    frosted = []
    for img in images:
        npimg = img.cpu().numpy()
        frost_noise = np.random.normal(0.7, 0.3, npimg.shape)
        frosted_img = npimg * (1 - frost_strength) + frost_strength * frost_noise
        frosted.append(torch.tensor(frosted_img))
    return torch.stack(frosted).clamp(-1, 1).float()

# =========================
# Evaluation helper
# =========================

def evaluate_attack(model, attack_fn, testloader):
    accuracies = []
    model.eval()

    print("\n🔥 Evaluating robustness...")

    with torch.no_grad():
        # Outer loop: severities 1–5
        for severity in tqdm(SEVERITY_LEVELS, desc="Severity levels", leave=True):
            correct = 0
            total = 0

            # Inner loop: iterate test set
            for images, labels in tqdm(
                testloader,
                desc=f"S{severity} batches",
                leave=False
            ):
                images = attack_fn(images, severity).to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = 100 * correct / total
            accuracies.append(acc)

    print("\n✔ Finished robustness evaluation.")
    return accuracies

# =========================
# Resolution visualisation helpers
# =========================

resolutions = [256, 64]
transforms_dict = {
    r: transforms.Compose([
        transforms.Resize((r, r)),
        transforms.ToTensor()
    ])
    for r in resolutions
}

def plot_same_images_with_labels(dataset, resolutions):
    n = len(dataset)
    fig, axs = plt.subplots(len(resolutions), n, figsize=(n * 3, len(resolutions) * 3))

    if len(resolutions) == 1:
        axs = axs[None, :]

    for j in range(n):
        img, label_idx = dataset[j]          # img is PIL
        label = dataset.dataset.classes[label_idx]
        for i, r in enumerate(resolutions):
            img_r = transforms_dict[r](img)
            img_np = img_r.permute(1, 2, 0).numpy()
            axs[i, j].imshow(img_np)
            axs[i, j].axis('off')
            if i == 0:
                axs[i, j].set_title(f"{label}", fontsize=8)
            if j == 0:
                axs[i, j].text(
                    -0.5, 0.5, f"{r}x{r}",
                    rotation=90, fontsize=8,
                    va='center', ha='center',
                    transform=axs[i, j].transAxes
                )
    plt.tight_layout()
    plt.show()

# =========================
# ViT factory
# =========================

def make_food_vit():
    return ViT(
        image_size = 64,
        patch_size = 8,
        num_classes = 101,
        dim = 256,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )

# =========================
# Main script
# =========================

def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Dataset & Dataloaders
    # =========================

    # root MUST directly contain images/ and meta/
    food_root = r"C:\Users\railt\Desktop\James\data"

    # Raw datasets for visualisation (PIL images)
    train_dataset_raw = datasets.Food101(root=food_root, split="train", transform=None)
    test_dataset_raw  = datasets.Food101(root=food_root, split="test",  transform=None)

    # Subset of raw dataset for resolution plots
    subset_train_idx = random.sample(range(len(train_dataset_raw)), 5)
    train_subset_raw = Subset(train_dataset_raw, subset_train_idx)

    # Plot same images with different resolutions
    plot_same_images_with_labels(train_subset_raw, resolutions)

    # Transform for model input: 64x64 + normalize to [-1, 1]
    transform_food101_64 = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Transformed datasets (tensors) for training / robustness / models
    train_dataset_food101 = datasets.Food101(
        root=food_root,
        split="train",
        transform=transform_food101_64,
        download=False
    )

    test_dataset_food101 = datasets.Food101(
        root=food_root,
        split="test",
        transform=transform_food101_64,
        download=False
    )

    train_loader_food101 = DataLoader(
        train_dataset_food101,
        batch_size=64,
        shuffle=True,
        num_workers=0   # safe now that we use if __name__ == "__main__"
    )

    test_loader_food101 = DataLoader(
        test_dataset_food101,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    print(f"Food-101 train samples: {len(train_dataset_food101)}")
    print(f"Food-101 test  samples: {len(test_dataset_food101)}")
    print(f"Number of classes: {len(train_dataset_food101.classes)}")

    # =========================
    # Build a small tensor batch for visualising corruptions
    # =========================

    subset_train_idx_robust = random.sample(range(len(train_dataset_food101)), 5)
    train_subset_robust = Subset(train_dataset_food101, subset_train_idx_robust)

    tensor_images = []
    labels_list  = []

    for img, label in train_subset_robust:
        tensor_images.append(img)      # img is now a 64×64 normalized tensor
        labels_list.append(label)

    images_64 = torch.stack(tensor_images)   # (5, 3, 64, 64)
    labels_64 = torch.tensor(labels_list)

    # 🔥 Using severity-based attacks (mid severity = 3)
    modifications = {
        "Original":      images_64,
        "Noise":         add_noise(images_64, severity=3),
        "Brightness":    change_brightness(images_64, severity=3),
        "Gaussian blur": defocus_blur(images_64, severity=3),
        "Pixelate":      pixelate(images_64, severity=3),
        "Fog":           add_fog(images_64, severity=3),
        "Frost":         frost(images_64, severity=3),
    }

    # 🎨 Visualisation of corrupted images
    for name, attacked_imgs in modifications.items():
        plt.figure(figsize=(10, 4))
        for i in range(len(attacked_imgs)):
            # un-normalise from [-1,1] to [0,1] for display
            img = attacked_imgs[i] * 0.5 + 0.5
            npimg = img.numpy().transpose(1, 2, 0)
            plt.subplot(2, 5, i + 1)
            plt.imshow(np.clip(npimg, 0, 1))
            plt.axis('off')
        plt.suptitle(f'{name} on Food-101 (64×64)', size=14)
        plt.show()

    # =========================
    # Load models
    # =========================

    ResNet18_food_PATH = r"C:/Users/railt/DST_Assessment_2/James/Resnet18_DIY/resnet18_food101_best.pth"
    Vit_food_PATH      = r"C:\Users\railt\DST_Assessment_2\James\SLViT_DIY/slvit_food101_best.pth"

    # ResNet18
    model_food_res = ResNet18(num_classes=101).to(device)
    state_food_res = torch.load(ResNet18_food_PATH, map_location=device)
    model_food_res.load_state_dict(state_food_res)
    model_food_res.eval()
    print("ResNet18 Food-101 model loaded successfully! 🍔🚀")

    # ViT
    model_vit_food = make_food_vit().to(device)
    state_vit_food = torch.load(Vit_food_PATH, map_location=device)
    model_vit_food.load_state_dict(state_vit_food)
    model_vit_food.eval()
    print("SLViT Food-101 BEST-ACC & BEST-LOSS loaded successfully! 🧠🍜")

    # =========================
    # Robustness evaluation
    # =========================

    attack_results_Food101_ResNet_best = {
        "Noise":       evaluate_attack(model_food_res, add_noise,        test_loader_food101),
        "Brightness":  evaluate_attack(model_food_res, change_brightness, test_loader_food101),
        "Blur":        evaluate_attack(model_food_res, defocus_blur,     test_loader_food101),
        "Pixelate":    evaluate_attack(model_food_res, pixelate,         test_loader_food101),
        "Fog":         evaluate_attack(model_food_res, add_fog,          test_loader_food101),
        "Frost":       evaluate_attack(model_food_res, frost,            test_loader_food101),
    }

    attack_results_Food101_ViT_best = {
        "Noise":       evaluate_attack(model_vit_food, add_noise,        test_loader_food101),
        "Brightness":  evaluate_attack(model_vit_food, change_brightness, test_loader_food101),
        "Blur":        evaluate_attack(model_vit_food, defocus_blur,     test_loader_food101),
        "Pixelate":    evaluate_attack(model_vit_food, pixelate,         test_loader_food101),
        "Fog":         evaluate_attack(model_vit_food, add_fog,          test_loader_food101),
        "Frost":       evaluate_attack(model_vit_food, frost,            test_loader_food101),
    }

    # =========================
    # Plot robustness curves
    # =========================

    attacks = list(attack_results_Food101_ResNet_best.keys())
    num_attacks = len(attacks)

    rows, cols = 2, 3  # 6 attacks → 2×3 grid
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
    axes = axes.flatten()
    fig.suptitle("Food-101 Robustness", fontsize=16, y=1.02)

    for idx, attack_name in enumerate(attacks):
        ax = axes[idx]

        # Get severity curves for both models
        acc_curve_resnet = attack_results_Food101_ResNet_best[attack_name]
        acc_curve_vit    = attack_results_Food101_ViT_best[attack_name]

        # Plot curves
        ax.plot(SEVERITY_LEVELS, acc_curve_resnet, marker='o', label='ResNet18')
        ax.plot(SEVERITY_LEVELS, acc_curve_vit,    marker='s', label='SLViT')

        ax.set_title(attack_name)
        ax.set_xlabel("Severity Level")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(SEVERITY_LEVELS)
        ax.grid(True)
        ax.legend(fontsize=8)

    # Hide any unused axes (in case)
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

