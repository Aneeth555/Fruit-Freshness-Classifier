import torch
from PIL import Image

from src.model import build_model
from src.transforms import val_transforms
from src.config import MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None   # global lazy-loaded model


def load_model():
    global _model
    if _model is None:
        print("🔄 Loading model...")
        model = build_model().to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()
        _model = model
    return _model


def predict_image(image: Image.Image):
    model = load_model()   # lazy load

    image = image.convert("RGB")
    tensor = val_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(tensor).item()   # Rotten probability

    if prob >= 0.6:
        label = "Avoid"
    elif prob >= 0.3:
        label = "Okay"
    else:
        label = "Fresh"

    return {
        "label": label,
        "rotten_probability": round(prob, 3)
    }
