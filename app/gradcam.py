import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Enable gradients ONLY for last conv block
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def save_gradients(self, grad):
        self.gradients = grad

    def generate(self, input_tensor):
        def forward_hook(module, input, output):
            self.activations = output
            output.register_hook(self.save_gradients)

        handle = self.model.layer4[-1].register_forward_hook(forward_hook)

        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()

        handle.remove()

        if self.gradients is None:
            raise RuntimeError("Grad-CAM failed: no gradients captured")

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam
