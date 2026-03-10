import json
import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.experiment import Experiment
from models.dataset import Dataset
from models.evaluation import Evaluation
from models.project import Project
from config import settings


class EvaluationService:
    async def evaluate(self, experiment: Experiment, db: AsyncSession) -> Evaluation:
        # Get dataset and project info
        dataset = await db.get(Dataset, experiment.dataset_id)
        project = await db.get(Project, experiment.project_id)
        task_type = project.task_type
        hp = json.loads(experiment.hyperparameters)
        resource_config = json.loads(experiment.resource_config) if experiment.resource_config else {}

        checkpoint_dir = settings.checkpoint_dir / experiment.id
        best_checkpoint = str(checkpoint_dir / "best_model.pt")

        # Run evaluation in thread
        result = await asyncio.to_thread(
            self._run_evaluation,
            task_type, experiment.architecture, experiment.pretrained,
            hp, resource_config, dataset.path, dataset.num_classes,
            best_checkpoint, dataset.annotation_format,
        )

        evaluation = Evaluation(
            experiment_id=experiment.id,
            metrics=json.dumps(result["metrics"]),
            confusion_matrix=json.dumps(result.get("confusion_matrix")),
            per_class_metrics=json.dumps(result.get("per_class_metrics")),
            best_checkpoint=best_checkpoint,
        )
        db.add(evaluation)
        await db.commit()
        await db.refresh(evaluation)
        return evaluation

    def _run_evaluation(
        self, task_type, architecture, pretrained, hp, resource_config,
        dataset_path, num_classes, checkpoint_path, annotation_format,
    ) -> dict:
        if task_type == "classification":
            from ml.evaluators.classification_evaluator import ClassificationEvaluator
            evaluator = ClassificationEvaluator(
                architecture=architecture,
                num_classes=num_classes,
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                hp=hp,
                resource_config=resource_config,
            )
        elif task_type == "detection":
            from ml.evaluators.detection_evaluator import DetectionEvaluator
            evaluator = DetectionEvaluator(
                architecture=architecture,
                num_classes=num_classes,
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                annotation_format=annotation_format,
                hp=hp,
                resource_config=resource_config,
            )
        elif task_type == "segmentation":
            from ml.evaluators.segmentation_evaluator import SegmentationEvaluator
            evaluator = SegmentationEvaluator(
                architecture=architecture,
                num_classes=num_classes,
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                hp=hp,
                resource_config=resource_config,
            )
        else:
            return {"metrics": {}}

        return evaluator.evaluate()
