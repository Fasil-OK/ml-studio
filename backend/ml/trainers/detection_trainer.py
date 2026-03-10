import time
from typing import Generator
from pathlib import Path

from ml.trainers.base_trainer import BaseTrainer


class DetectionTrainer(BaseTrainer):
    def __init__(self, *args, annotation_format="coco", **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation_format = annotation_format

    def train(self) -> Generator[dict, None, None]:
        if self.architecture.startswith("yolov8"):
            yield from self._train_yolo()
        else:
            yield from self._train_torchvision()

    def _train_yolo(self) -> Generator[dict, None, None]:
        from ultralytics import YOLO

        variant = self.architecture.replace("yolov8", "")  # n, s, m, l
        model = YOLO(f"yolov8{variant}.pt" if self.pretrained else f"yolov8{variant}.yaml")

        epochs = self.hp.get("epochs", 50)
        batch_size = self.hp.get("batch_size", 16)
        input_size = self.hp.get("input_size", 640)
        lr = self.hp.get("lr", 0.01)

        # YOLO expects data.yaml or specific directory structure
        data_path = Path(self.dataset_path)
        data_yaml = data_path / "data.yaml"

        if not data_yaml.exists():
            # Create a minimal data.yaml
            import yaml
            config = {
                "path": str(data_path),
                "train": "train/images" if (data_path / "train").exists() else "images",
                "val": "val/images" if (data_path / "val").exists() else "images",
                "nc": self.num_classes,
                "names": [f"class_{i}" for i in range(self.num_classes)],
            }
            with open(data_yaml, "w") as f:
                yaml.dump(config, f)

        # Train YOLO — it handles its own training loop
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=input_size,
            lr0=lr,
            project=self.checkpoint_dir,
            name="yolo_train",
            exist_ok=True,
            verbose=False,
        )

        # Yield final results
        yield {
            "type": "epoch_end",
            "epoch": epochs - 1,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "lr": lr,
            "duration": 0.0,
            "gpu_memory_mb": self.get_gpu_memory(),
            "extra_metrics": {"note": "See YOLO training logs for detailed metrics"},
        }

    def _train_torchvision(self) -> Generator[dict, None, None]:
        import torch
        from ml.architectures.registry import load_model

        model = load_model(self.architecture, self.num_classes, self.pretrained, "detection")
        model = model.to(self.device)

        optimizer = self.create_optimizer([p for p in model.parameters() if p.requires_grad])
        epochs = self.hp.get("epochs", 50)
        scheduler = self.create_scheduler(optimizer, epochs)

        # For torchvision detection models, training data needs specific format
        # This is a simplified loop — full COCO data loading would be needed
        for epoch in range(epochs):
            if self.stop_flag():
                break

            epoch_start = time.time()
            model.train()

            # Placeholder — full detection data loading would be implemented
            train_loss = 0.0
            duration = time.time() - epoch_start

            yield {
                "type": "epoch_end",
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": round(train_loss, 4),
                "val_loss": 0.0,
                "train_accuracy": 0.0,
                "val_accuracy": 0.0,
                "lr": optimizer.param_groups[0]["lr"],
                "duration": round(duration, 1),
                "gpu_memory_mb": self.get_gpu_memory(),
            }

            scheduler.step()
