import time
from typing import Generator

import torch
import torch.nn as nn

from ml.trainers.base_trainer import BaseTrainer
from ml.architectures.registry import load_model
from ml.data.classification_dataset import create_data_loaders


class ClassificationTrainer(BaseTrainer):
    def train(self) -> Generator[dict, None, None]:
        model = load_model(self.architecture, self.num_classes, self.pretrained, "classification")
        model = model.to(self.device)

        optimizer = self.create_optimizer(model.parameters())
        epochs = self.hp.get("epochs", 50)
        scheduler = self.create_scheduler(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler() if self.mixed_precision else None

        input_size = self.hp.get("input_size", 224)
        batch_size = self.hp.get("batch_size", 32)
        augmentation = self.hp.get("augmentation", "light")
        num_workers = self.resource_config.get("num_workers", 2)

        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            self.dataset_path, input_size, batch_size, augmentation, num_workers,
        )

        best_val_acc = 0.0
        patience = self.hp.get("early_stopping_patience", 10)
        patience_counter = 0

        for epoch in range(epochs):
            if self.stop_flag():
                break

            epoch_start = time.time()

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                if self.stop_flag():
                    break
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                if self.mixed_precision and scaler:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                # Batch update every 10 batches
                if batch_idx % 10 == 0:
                    yield {
                        "type": "batch_update",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": len(train_loader),
                        "loss": loss.item(),
                    }

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # Validation phase
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            if val_loader:
                model.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * images.size(0)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                val_loss /= max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)
            else:
                val_acc = train_acc
                val_loss = train_loss

            # Scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(model, optimizer, epoch, val_acc, class_names, "classification")
                patience_counter = 0
            else:
                patience_counter += 1

            duration = time.time() - epoch_start

            yield {
                "type": "epoch_end",
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "duration": round(duration, 1),
                "gpu_memory_mb": self.get_gpu_memory(),
                "best_val_accuracy": round(best_val_acc, 4),
            }

            if patience_counter >= patience:
                break
