import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.experiment import Experiment
from models.training_run import TrainingMetric
from schemas.training import TrainingStatusResponse, TrainingMetricResponse
from services.training_service import TrainingService

router = APIRouter()
training_service = TrainingService()


@router.post("/experiments/{experiment_id}/train")
async def start_training(experiment_id: str, db: AsyncSession = Depends(get_db)):
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found")
    if experiment.status == "running":
        raise HTTPException(400, "Training already running")

    experiment.status = "running"
    await db.commit()

    await training_service.start(experiment_id, db)
    return {"status": "started", "experiment_id": experiment_id}


@router.post("/experiments/{experiment_id}/train/stop")
async def stop_training(experiment_id: str, db: AsyncSession = Depends(get_db)):
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found")

    training_service.stop(experiment_id)
    experiment.status = "stopped"
    await db.commit()
    return {"status": "stopped"}


@router.get("/experiments/{experiment_id}/train", response_model=TrainingStatusResponse)
async def get_training_status(experiment_id: str, db: AsyncSession = Depends(get_db)):
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found")

    result = await db.execute(
        select(TrainingMetric)
        .where(TrainingMetric.experiment_id == experiment_id)
        .order_by(TrainingMetric.epoch)
    )
    metrics = result.scalars().all()

    hp = json.loads(experiment.hyperparameters)
    return TrainingStatusResponse(
        experiment_id=experiment_id,
        status=experiment.status,
        current_epoch=metrics[-1].epoch if metrics else None,
        total_epochs=hp.get("epochs"),
        metrics=[_parse_metric(m) for m in metrics],
    )


def _parse_metric(m: TrainingMetric) -> dict:
    return {
        "epoch": m.epoch,
        "train_loss": m.train_loss,
        "train_accuracy": m.train_accuracy,
        "val_loss": m.val_loss,
        "val_accuracy": m.val_accuracy,
        "learning_rate": m.learning_rate,
        "epoch_duration": m.epoch_duration,
        "gpu_memory_used": m.gpu_memory_used,
        "extra_metrics": json.loads(m.extra_metrics) if m.extra_metrics else None,
        "timestamp": m.timestamp,
    }
