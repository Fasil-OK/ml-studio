import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Detect classes from subdirectories
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.classes = classes

        for class_name in classes:
            class_dir = self.root / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(input_size: int = 224, augmentation: str = "light", is_train: bool = True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if not is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])

    if augmentation == "none":
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])
    elif augmentation == "light":
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # heavy
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            normalize,
        ])


def create_data_loaders(
    dataset_path: str,
    input_size: int = 224,
    batch_size: int = 32,
    augmentation: str = "light",
    num_workers: int = 2,
    val_split: float = 0.15,
    test_split: float = 0.15,
):
    root = Path(dataset_path)

    # Check for pre-split structure
    has_splits = (root / "train").exists()

    if has_splits:
        train_transform = get_transforms(input_size, augmentation, is_train=True)
        val_transform = get_transforms(input_size, augmentation, is_train=False)

        train_dataset = ImageFolderDataset(str(root / "train"), transform=train_transform)
        val_dir = root / "val" if (root / "val").exists() else root / "validation"
        val_dataset = ImageFolderDataset(str(val_dir), transform=val_transform) if val_dir.exists() else None
        test_dataset = ImageFolderDataset(str(root / "test"), transform=val_transform) if (root / "test").exists() else None

        class_names = train_dataset.classes
    else:
        # Single directory — split ourselves
        full_dataset = ImageFolderDataset(root, transform=None)
        class_names = full_dataset.classes

        n = len(full_dataset)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test

        train_subset, val_subset, test_subset = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        train_transform = get_transforms(input_size, augmentation, is_train=True)
        val_transform = get_transforms(input_size, augmentation, is_train=False)

        train_dataset = TransformSubset(train_subset, train_transform)
        val_dataset = TransformSubset(val_subset, val_transform)
        test_dataset = TransformSubset(test_subset, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    ) if test_dataset else None

    return train_loader, val_loader, test_loader, class_names


class TransformSubset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
