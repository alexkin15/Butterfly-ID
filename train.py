import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ========================
# Config
# ========================
class Config:
    # Paths to dataset and output folder (change these if needed)
    data_dir = "/content/drive/MyDrive/butterfly-id/data/images"
    out_dir  = "/content/drive/MyDrive/butterfly-id/output"
    
    # Training hyperparameters
    batch_size = 32
    num_workers = 2
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 20
    img_size = 224

cfg = Config()
cfg.save_interval = 900  # save model every 15 minutes
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

BEST_PATH = Path(cfg.out_dir) / "best_model_state.pt"  # always keeps latest/best model
CKPT_PATH = Path(cfg.out_dir) / "checkpoint.pth"       # checkpoint to resume training

# Select GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Training on:", device)

# ========================
# Dataset + Auto Class Detection
# ========================
# Auto-detect number of classes from training set
train_ds = datasets.ImageFolder(cfg.data_dir + "/train")
cfg.num_classes = len(train_ds.classes)
print(f"[INFO] Detected {cfg.num_classes} species")

# ========================
# Model
# ========================
# Load ResNet50 pre-trained on ImageNet, replace the last layer with new classifier
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, cfg.num_classes)
model = model.to(device)

# Loss function + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

start_epoch = 0
best_acc = -1.0   # initialize with -1 so the model is saved in the first epoch

# ========================
# Data Augmentation (train) + Normalization (val)
# ========================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # add variation
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_ds = datasets.ImageFolder(Path(cfg.data_dir) / "train", transform=train_tf)
val_ds   = datasets.ImageFolder(Path(cfg.data_dir) / "val", transform=val_tf)

# Wrap datasets in DataLoader for batching
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

# ========================
# Resume from checkpoint
# ========================
# If a checkpoint exists, resume training from there
if CKPT_PATH.exists():
    print(f"[INFO] Resuming from checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1  # continue from next epoch
    best_acc = ckpt["best_acc"]

# ========================
# Training loop
# ========================
start_time = time.time()
last_save_time = start_time

for epoch in range(start_epoch, cfg.epochs):
    model.train()
    running_loss, total, correct = 0.0, 0, 0

    # tqdm adds a progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward + Backward pass
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track training accuracy & loss
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / total
        pbar.set_postfix({"Train Loss": f"{train_loss:.4f}", "Train Acc": f"{train_acc:.4f}"})

        # ⏳ Save checkpoint every 15 minutes
        if time.time() - last_save_time > cfg.save_interval:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": best_acc,
            }, CKPT_PATH)

            # Also update best model (acts as latest backup)
            torch.save(model.state_dict(), BEST_PATH)
            print(f"[INFO] Autosaved checkpoint + best model at epoch {epoch}")
            last_save_time = time.time()

    # ========================
    # Validation loop
    # ========================
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0

    # Log metrics for this epoch
    print(f"✅ Epoch [{epoch+1}/{cfg.epochs}] "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Save if validation accuracy improves
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_PATH)
        print(f"[INFO] Saved BEST model (Val Acc={best_acc:.4f})")

    # Always save checkpoint at the end of each epoch
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
    }, CKPT_PATH)

    # Also update best model at the end of epoch (safety net)
    torch.save(model.state_dict(), BEST_PATH)

print("[INFO] Training finished. Best Val Acc=", best_acc)
