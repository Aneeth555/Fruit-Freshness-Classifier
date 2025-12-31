import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.model import build_model
from src.transforms import val_transforms
from src.config import MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = build_model().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Enable gradients for last conv block
for p in model.layer4.parameters():
    p.requires_grad = True

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None

    def generate(self, x):
        def hook(module, inp, out):
            self.activations = out
            out.retain_grad()

        h = model.layer4[-1].register_forward_hook(hook)

        logits = self.model(x)
        score = logits.squeeze()

        self.model.zero_grad()
        score.backward()

        h.remove()

        grads = self.activations.grad
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

def run_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transforms()(image).unsqueeze(0).to(device)

    cam = GradCAM(model).generate(input_tensor)

    img = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(cam, cmap="jet"); plt.title("Grad-CAM"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_gradcam("data/test/c_f001.png")
