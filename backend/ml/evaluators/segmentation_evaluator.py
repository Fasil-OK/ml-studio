import torch
import numpy as np

from ml.architectures.registry import load_model
from ml.data.segmentation_dataset import create_segmentation_loaders


class SegmentationEvaluator:
    def __init__(self, architecture, num_classes, checkpoint_path, dataset_path, hp, resource_config):
        self.architecture = architecture
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.hp = hp
        self.device = resource_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self) -> dict:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model = load_model(self.architecture, self.num_classes, pretrained=False, task_type="segmentation")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        input_size = self.hp.get("input_size", 256)
        batch_size = self.hp.get("batch_size", 16)

        _, _, test_loader = create_segmentation_loaders(
            self.dataset_path, input_size, batch_size,
        )

        total_iou = np.zeros(self.num_classes)
        total_count = np.zeros(self.num_classes)
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                preds = outputs.argmax(dim=1)

                total_correct += (preds == masks).sum().item()
                total_pixels += masks.numel()

                for c in range(self.num_classes):
                    pred_c = (preds == c)
                    target_c = (masks == c)
                    intersection = (pred_c & target_c).sum().item()
                    union = (pred_c | target_c).sum().item()
                    if union > 0:
                        total_iou[c] += intersection / union
                        total_count[c] += 1

        per_class_iou = []
        for c in range(self.num_classes):
            iou = total_iou[c] / max(total_count[c], 1)
            per_class_iou.append({
                "class": f"class_{c}",
                "iou": round(float(iou), 4),
            })

        miou = float(np.mean(total_iou / np.maximum(total_count, 1)))
        pixel_acc = total_correct / max(total_pixels, 1)

        return {
            "metrics": {
                "mIoU": round(miou, 4),
                "pixel_accuracy": round(pixel_acc, 4),
            },
            "confusion_matrix": None,
            "per_class_metrics": per_class_iou,
        }
