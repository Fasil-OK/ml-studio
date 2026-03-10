import json
import time
import asyncio
from io import BytesIO

import torch
from PIL import Image
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from models.experiment import Experiment
from models.dataset import Dataset
from models.project import Project
from config import settings


class InferenceService:
    _model_cache: dict[str, tuple] = {}  # experiment_id -> (model, transforms, class_names, task_type)

    async def predict(self, experiment: Experiment, image: UploadFile) -> dict:
        image_bytes = await image.read()

        result = await asyncio.to_thread(
            self._run_prediction, experiment, image_bytes
        )
        return result

    def _run_prediction(self, experiment: Experiment, image_bytes: bytes) -> dict:
        import torch
        from torchvision import transforms

        hp = json.loads(experiment.hyperparameters)
        resource_config = json.loads(experiment.resource_config) if experiment.resource_config else {}
        device = resource_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = settings.checkpoint_dir / experiment.id / "best_model.pt"

        if experiment.id not in self._model_cache:
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            model = checkpoint.get("model")
            class_names = checkpoint.get("class_names", [])
            task_type = checkpoint.get("task_type", "classification")

            if model is None:
                # Load from architecture
                from ml.architectures.registry import load_model
                model = load_model(
                    experiment.architecture, checkpoint.get("num_classes", 10),
                    pretrained=False, task_type=task_type
                )
                model.load_state_dict(checkpoint["model_state_dict"])

            model = model.to(device)
            model.eval()

            input_size = hp.get("input_size", 224)
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self._model_cache[experiment.id] = (model, transform, class_names, task_type)

        model, transform, class_names, task_type = self._model_cache[experiment.id]

        # Process image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        start_time = time.time()

        if task_type == "classification":
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_k = min(5, len(class_names))
            values, indices = probs.topk(top_k)
            predictions = [
                {"class": class_names[idx] if idx < len(class_names) else f"class_{idx}",
                 "confidence": round(val.item() * 100, 2)}
                for val, idx in zip(values, indices)
            ]
        else:
            predictions = [{"class": "prediction", "confidence": 0.0}]

        processing_time = round((time.time() - start_time) * 1000, 1)
        return {
            "predictions": predictions,
            "processing_time_ms": processing_time,
            "task_type": task_type,
        }
