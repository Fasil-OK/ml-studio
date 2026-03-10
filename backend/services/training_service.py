import asyncio
import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import async_session
from models.experiment import Experiment
from models.dataset import Dataset
from models.training_run import TrainingMetric
from ws.manager import ws_manager

logger = logging.getLogger(__name__)


class TrainingService:
    _active_tasks: dict[str, asyncio.Task] = {}
    _stop_flags: dict[str, bool] = {}

    async def start(self, experiment_id: str, db: AsyncSession):
        experiment = await db.get(Experiment, experiment_id)
        if not experiment:
            return

        hp = json.loads(experiment.hyperparameters)
        resource_config = json.loads(experiment.resource_config) if experiment.resource_config else {}

        # Get dataset
        result = await db.execute(select(Dataset).where(Dataset.id == experiment.dataset_id))
        dataset = result.scalars().first()

        self._stop_flags[experiment_id] = False

        # Determine task type from project
        from models.project import Project
        project = await db.get(Project, experiment.project_id)
        task_type = project.task_type

        # Launch training in background
        task = asyncio.create_task(
            self._run_training(experiment_id, task_type, experiment.architecture,
                             experiment.pretrained, hp, resource_config,
                             dataset.path, dataset.num_classes, dataset.annotation_format)
        )
        self._active_tasks[experiment_id] = task

    def stop(self, experiment_id: str):
        self._stop_flags[experiment_id] = True

    async def _run_training(
        self, experiment_id: str, task_type: str, architecture: str,
        pretrained: bool, hp: dict, resource_config: dict,
        dataset_path: str, num_classes: int, annotation_format: str
    ):
        loop = asyncio.get_event_loop()
        try:
            if task_type == "classification":
                from ml.trainers.classification_trainer import ClassificationTrainer
                trainer = ClassificationTrainer(
                    experiment_id=experiment_id,
                    architecture=architecture,
                    pretrained=pretrained,
                    hyperparameters=hp,
                    resource_config=resource_config,
                    dataset_path=dataset_path,
                    num_classes=num_classes,
                    checkpoint_dir=str(settings.checkpoint_dir / experiment_id),
                    stop_flag=lambda: self._stop_flags.get(experiment_id, False),
                )
            elif task_type == "detection":
                from ml.trainers.detection_trainer import DetectionTrainer
                trainer = DetectionTrainer(
                    experiment_id=experiment_id,
                    architecture=architecture,
                    pretrained=pretrained,
                    hyperparameters=hp,
                    resource_config=resource_config,
                    dataset_path=dataset_path,
                    num_classes=num_classes,
                    annotation_format=annotation_format,
                    checkpoint_dir=str(settings.checkpoint_dir / experiment_id),
                    stop_flag=lambda: self._stop_flags.get(experiment_id, False),
                )
            elif task_type == "segmentation":
                from ml.trainers.segmentation_trainer import SegmentationTrainer
                trainer = SegmentationTrainer(
                    experiment_id=experiment_id,
                    architecture=architecture,
                    pretrained=pretrained,
                    hyperparameters=hp,
                    resource_config=resource_config,
                    dataset_path=dataset_path,
                    num_classes=num_classes,
                    checkpoint_dir=str(settings.checkpoint_dir / experiment_id),
                    stop_flag=lambda: self._stop_flags.get(experiment_id, False),
                )
            else:
                return

            def _train():
                for update in trainer.train():
                    asyncio.run_coroutine_threadsafe(
                        self._handle_update(experiment_id, update),
                        loop,
                    ).result(timeout=5)

            await asyncio.to_thread(_train)

            # Mark as completed
            async with async_session() as db:
                exp = await db.get(Experiment, experiment_id)
                if exp and exp.status == "running":
                    exp.status = "completed"
                    await db.commit()

            await ws_manager.broadcast(f"training:{experiment_id}", {
                "type": "training_complete",
            })

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            async with async_session() as db:
                exp = await db.get(Experiment, experiment_id)
                if exp:
                    exp.status = "failed"
                    await db.commit()
            await ws_manager.broadcast(f"training:{experiment_id}", {
                "type": "training_failed",
                "error": str(e),
            })
        finally:
            self._active_tasks.pop(experiment_id, None)
            self._stop_flags.pop(experiment_id, None)

    async def _handle_update(self, experiment_id: str, update: dict):
        # Save metric to DB
        if update.get("type") == "epoch_end":
            async with async_session() as db:
                metric = TrainingMetric(
                    experiment_id=experiment_id,
                    epoch=update["epoch"],
                    train_loss=update.get("train_loss"),
                    train_accuracy=update.get("train_accuracy"),
                    val_loss=update.get("val_loss"),
                    val_accuracy=update.get("val_accuracy"),
                    learning_rate=update.get("lr"),
                    epoch_duration=update.get("duration"),
                    gpu_memory_used=update.get("gpu_memory_mb"),
                    extra_metrics=json.dumps(update.get("extra_metrics")) if update.get("extra_metrics") else None,
                )
                db.add(metric)
                await db.commit()

        # Broadcast via WebSocket
        await ws_manager.broadcast(f"training:{experiment_id}", update)
