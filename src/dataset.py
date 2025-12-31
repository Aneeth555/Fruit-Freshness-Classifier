from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from src.config import DATASET_PATH
from src.transforms import train_transforms, val_transforms

def load_datasets():
    full_dataset = ImageFolder(DATASET_PATH, transform=train_transforms())

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transforms()

    return train_ds, val_ds
