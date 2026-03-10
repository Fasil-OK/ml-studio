import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from ml.architectures.registry import load_model
from ml.data.classification_dataset import create_data_loaders


class ClassificationEvaluator:
    def __init__(self, architecture, num_classes, checkpoint_path, dataset_path, hp, resource_config):
        self.architecture = architecture
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.hp = hp
        self.device = resource_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self) -> dict:
        # Load model
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model = load_model(self.architecture, self.num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        class_names = checkpoint.get("class_names", [f"class_{i}" for i in range(self.num_classes)])
        input_size = self.hp.get("input_size", 224)
        batch_size = self.hp.get("batch_size", 32)

        _, _, test_loader, _ = create_data_loaders(
            self.dataset_path, input_size, batch_size, "none", num_workers=2,
        )

        # If no test loader, use val loader
        if test_loader is None:
            _, test_loader, _, _ = create_data_loaders(
                self.dataset_path, input_size, batch_size, "none", num_workers=2,
            )

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds).tolist()

        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        per_class = []
        for name in class_names:
            if name in report:
                per_class.append({
                    "class": name,
                    "precision": round(report[name]["precision"], 4),
                    "recall": round(report[name]["recall"], 4),
                    "f1": round(report[name]["f1-score"], 4),
                    "support": int(report[name]["support"]),
                })

        return {
            "metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            },
            "confusion_matrix": cm,
            "per_class_metrics": per_class,
        }
