class DetectionEvaluator:
    def __init__(self, architecture, num_classes, checkpoint_path, dataset_path,
                 annotation_format, hp, resource_config):
        self.architecture = architecture
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.annotation_format = annotation_format
        self.hp = hp
        self.resource_config = resource_config

    def evaluate(self) -> dict:
        if self.architecture.startswith("yolov8"):
            return self._evaluate_yolo()
        return self._evaluate_torchvision()

    def _evaluate_yolo(self) -> dict:
        from ultralytics import YOLO
        from pathlib import Path

        # Find the best YOLO model
        checkpoint_dir = Path(self.checkpoint_path).parent
        best_path = checkpoint_dir / "yolo_train" / "weights" / "best.pt"
        if not best_path.exists():
            best_path = self.checkpoint_path

        model = YOLO(str(best_path))
        data_yaml = Path(self.dataset_path) / "data.yaml"

        results = model.val(data=str(data_yaml), verbose=False)

        metrics = {
            "mAP50": round(float(results.box.map50), 4) if hasattr(results.box, "map50") else 0.0,
            "mAP50_95": round(float(results.box.map), 4) if hasattr(results.box, "map") else 0.0,
        }

        return {
            "metrics": metrics,
            "confusion_matrix": None,
            "per_class_metrics": [],
        }

    def _evaluate_torchvision(self) -> dict:
        return {
            "metrics": {"mAP50": 0.0, "mAP50_95": 0.0},
            "confusion_matrix": None,
            "per_class_metrics": [],
        }
