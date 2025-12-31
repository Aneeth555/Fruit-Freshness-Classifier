import os
import torch
from PIL import Image

from src.model import build_model
from src.transforms import val_transforms
from src.config import MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = build_model().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

AVOID_THRESHOLD = 0.65
OKAY_THRESHOLD = 0.40

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = val_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        prob = torch.sigmoid(logits).item()

    if prob >= AVOID_THRESHOLD:
        label = "Avoid"
    elif prob >= OKAY_THRESHOLD:
        label = "Okay"
    else:
        label = "Fresh"

    return label, prob

def predict_folder(folder):
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, file)
            label, prob = predict_image(path)
            print(f"{file:25} → {label:5} | Rotten prob: {prob:.3f}")

if __name__ == "__main__":
    predict_folder("data/test")
