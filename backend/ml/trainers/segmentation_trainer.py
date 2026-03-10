import time
from typing import Generator

import torch
import torch.nn as nn

from ml.trainers.base_trainer import BaseTrainer
from ml.architectures.registry import load_model
from ml.data.segmentation_dataset import create_segmentation_loaders


class SegmentationTrainer(BaseTrainer):
    def train(self) -> Generator[dict, None, None]:
        model = load_model(self.architecture, self.num_classes, self.pretrained, "segmentation")
        model = model.to(self.device)

        optimizer = self.create_optimizer(model.parameters())
        epochs = self.hp.get("epochs", 50)
        scheduler = self.create_scheduler(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler() if self.mixed_precision else None

        input_size = self.hp.get("input_size", 256)
        batch_size = self.hp.get("batch_size", 16)
        num_workers = self.resource_config.get("num_workers", 2)

        train_loader, val_loader, test_loader = create_segmentation_loaders(
            self.dataset_path, input_size, batch_size, num_workers,
        )

        best_val_iou = 0.0

        for epoch in range(epochs):
            if self.stop_flag():
                break

            epoch_start = time.time()

            # Training
            model.train()
            train_loss = 0.0
            train_total = 0

            for batch_idx, (images, masks) in enumerate(train_loader):
                if self.stop_flag():
                    break
                images, masks = images.to(self.device), masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                # Handle torchvision segmentation models that return dict
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_total += images.size(0)

                if batch_idx % 10 == 0:
                    yield {
                        "type": "batch_update",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": len(train_loader),
                        "loss": loss.item(),
                    }

            train_loss /= max(train_total, 1)

            # Validation
            val_loss = 0.0
            val_total = 0
            val_iou = 0.0
            iou_count = 0

            if val_loader:
                model.eval()
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(self.device), masks.to(self.device)
                        outputs = model(images)
                        if isinstance(outputs, dict):
                            outputs = outputs["out"]
                        loss = criterion(outputs, masks)
                        val_loss += loss.item() * images.size(0)
                        val_total += images.size(0)

                        # Compute IoU
                        preds = outputs.argmax(dim=1)
                        for c in range(self.num_classes):
                            pred_c = (preds == c)
                            target_c = (masks == c)
                            intersection = (pred_c & target_c).sum().item()
                            union = (pred_c | target_c).sum().item()
                            if union > 0:
                                val_iou += intersection / union
                                iou_count += 1

                val_loss /= max(val_total, 1)
                val_iou = val_iou / max(iou_count, 1)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                self.save_checkpoint(model, optimizer, epoch, val_iou, task_type="segmentation")

            duration = time.time() - epoch_start

            yield {
                "type": "epoch_end",
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": round(train_loss, 4),
                "train_accuracy": 0.0,
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_iou, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "duration": round(duration, 1),
                "gpu_memory_mb": self.get_gpu_memory(),
                "extra_metrics": {"mIoU": round(val_iou, 4)},
            }
