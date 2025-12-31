import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def build_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Classifier head (NO sigmoid)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)   # 🔥 logits
    )

    return model
