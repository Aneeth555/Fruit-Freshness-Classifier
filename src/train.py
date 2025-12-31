import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import load_datasets
from src.model import build_model
from src.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_ds, val_ds = load_datasets()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = build_model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

start_epoch = 0
best_acc = 0.0

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]

def save_checkpoint(epoch, acc):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": acc
    }, MODEL_PATH)

def train_one_epoch(epoch):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs = imgs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(imgs).squeeze()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

for epoch in range(start_epoch, EPOCHS):
    loss = train_one_epoch(epoch)
    acc = validate()

    print(f"Loss: {loss:.4f} | Val Acc: {acc:.4f}")
    save_checkpoint(epoch, acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("⭐ Best model saved")

print("✅ Training complete")
