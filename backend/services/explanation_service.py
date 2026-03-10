import json
import asyncio
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.experiment import Experiment
from models.dataset import Dataset
from models.project import Project
from config import settings


class ExplanationService:
    async def explain(self, experiment: Experiment, image: UploadFile, method: str, db: AsyncSession) -> dict:
        project = await db.get(Project, experiment.project_id)
        task_type = project.task_type

        # Only support explainability for classification (GradCAM etc.)
        if task_type != "classification":
            return {"error": "Explainability currently supports classification models only."}

        hp = json.loads(experiment.hyperparameters)
        resource_config = json.loads(experiment.resource_config) if experiment.resource_config else {}

        checkpoint_path = str(settings.checkpoint_dir / experiment.id / "best_model.pt")
        explanation_id = str(uuid.uuid4())
        output_dir = settings.explanation_dir / experiment.id / explanation_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded image
        image_bytes = await image.read()
        input_path = output_dir / "input.png"
        with open(input_path, "wb") as f:
            f.write(image_bytes)

        # Get class names from dataset
        ds_result = await db.execute(select(Dataset).where(Dataset.id == experiment.dataset_id))
        ds = ds_result.scalars().first()
        class_names = json.loads(ds.class_names or "[]")

        result = await asyncio.to_thread(
            self._generate_explanation,
            experiment.architecture,
            class_names,
            hp,
            resource_config,
            checkpoint_path,
            str(input_path),
            method,
            str(output_dir),
        )

        return {
            "explanation_id": explanation_id,
            "method": method,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "overlay_path": f"/storage/explanations/{experiment.id}/{explanation_id}/overlay.png",
            "heatmap_path": f"/storage/explanations/{experiment.id}/{explanation_id}/heatmap.png",
        }

    def _generate_explanation(
        self, architecture, class_names, hp, resource_config,
        checkpoint_path, image_path, method, output_dir,
    ) -> dict:
        from ml.explainer import Explainer
        explainer = Explainer(
            architecture=architecture,
            num_classes=len(class_names),
            checkpoint_path=checkpoint_path,
            class_names=class_names,
            device=resource_config.get("device", "cuda"),
        )
        return explainer.explain(image_path, method, output_dir)

    def list_explanations(self, experiment_id: str) -> list[dict]:
        exp_dir = settings.explanation_dir / experiment_id
        if not exp_dir.exists():
            return []
        results = []
        for sub in sorted(exp_dir.iterdir()):
            if sub.is_dir():
                results.append({
                    "explanation_id": sub.name,
                    "overlay_path": f"/storage/explanations/{experiment_id}/{sub.name}/overlay.png",
                    "heatmap_path": f"/storage/explanations/{experiment_id}/{sub.name}/heatmap.png",
                    "input_path": f"/storage/explanations/{experiment_id}/{sub.name}/input.png",
                })
        return results
