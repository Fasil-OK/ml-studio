import os
import time
from typing import Generator
from pathlib import Path

import torch


class BaseTrainer:
    def __init__(
        self, experiment_id, architecture, pretrained, hyperparameters,
        resource_config, dataset_path, num_classes, checkpoint_dir, stop_flag,
    ):
        self.experiment_id = experiment_id
        self.architecture = architecture
        self.pretrained = pretrained
        self.hp = hyperparameters
        self.resource_config = resource_config
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.stop_flag = stop_flag

        self.device = resource_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = resource_config.get("mixed_precision", False)

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def create_optimizer(self, params):
        name = self.hp.get("optimizer", "adam").lower()
        lr = self.hp.get("lr", 1e-3)
        weight_decay = self.hp.get("weight_decay", 1e-4)

        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def create_scheduler(self, optimizer, total_epochs):
        name = self.hp.get("scheduler", "cosine").lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(total_epochs // 3, 1), gamma=0.1)
        elif name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    def get_gpu_memory(self):
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1e6, 1)
        return 0.0

    def save_checkpoint(self, model, optimizer, epoch, val_metric, class_names=None, task_type="classification"):
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_metric": val_metric,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "class_names": class_names or [],
            "task_type": task_type,
            "hyperparameters": self.hp,
        }, path)

    def train(self) -> Generator[dict, None, None]:
        raise NotImplementedError
