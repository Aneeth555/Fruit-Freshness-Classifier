import torch
import numpy as np
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.dataset import load_datasets
from src.model import build_model
from src.config import MODEL_PATH, BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ LOAD MODEL ------------------
model = build_model().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ------------------ LOAD VALIDATION DATA ------------------
_, val_ds = load_datasets()
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

y_true = []
y_probs = []

# ------------------ COLLECT PROBS ------------------
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.cpu().numpy()

        logits = model(imgs).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

        y_true.extend(labels)
        y_probs.extend(probs)

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# ------------------ ROC CALIBRATION ------------------
fpr, tpr, thresholds = roc_curve(y_true, y_probs)

# Youden’s J statistic
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)

best_threshold = thresholds[best_idx]

# Define uncertainty zone
OKAY_LOW = best_threshold - 0.15
OKAY_HIGH = best_threshold + 0.15

# Clamp values
OKAY_LOW = max(0.0, OKAY_LOW)
OKAY_HIGH = min(1.0, OKAY_HIGH)

# ------------------ SAVE THRESHOLDS ------------------
np.save("calibration.npy", {
    "avoid": float(best_threshold),
    "okay_low": float(OKAY_LOW),
    "okay_high": float(OKAY_HIGH)
})

print("\n✅ Calibration complete")
print(f"Avoid threshold (Rotten): {best_threshold:.3f}")
print(f"Okay zone: {OKAY_LOW:.3f} – {OKAY_HIGH:.3f}")
