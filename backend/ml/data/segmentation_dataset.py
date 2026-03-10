from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, input_size: int = 256, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.input_size = input_size
        self.transform = transform

        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name
        if not mask_path.exists():
            mask_path = self.masks_dir / (img_path.stem + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)

        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        return img_tensor, mask_tensor


def create_segmentation_loaders(
    dataset_path: str, input_size: int = 256, batch_size: int = 16, num_workers: int = 2
):
    root = Path(dataset_path)
    images_dir = root / "images"
    masks_dir = root / "masks"

    if not images_dir.exists():
        images_dir = root
    if not masks_dir.exists():
        masks_dir = root

    dataset = SegmentationDataset(str(images_dir), str(masks_dir), input_size)

    n = len(dataset)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
