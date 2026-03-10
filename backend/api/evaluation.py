import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.experiment import Experiment
from models.evaluation import Evaluation as EvaluationModel
from schemas.evaluation import EvaluationResponse
from services.evaluation_service import EvaluationService

router = APIRouter()


@router.post("/experiments/{experiment_id}/evaluate")
async def run_evaluation(experiment_id: str, db: AsyncSession = Depends(get_db)):
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found")
    if experiment.status not in ("completed", "stopped"):
        raise HTTPException(400, "Training must be completed before evaluation")

    service = EvaluationService()
    evaluation = await service.evaluate(experiment, db)
    return _parse_evaluation(evaluation)


@router.get("/experiments/{experiment_id}/evaluation")
async def get_evaluation(experiment_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(EvaluationModel)
        .where(EvaluationModel.experiment_id == experiment_id)
        .order_by(EvaluationModel.created_at.desc())
    )
    evaluation = result.scalars().first()
    if not evaluation:
        return None
    return _parse_evaluation(evaluation)


def _parse_evaluation(e: EvaluationModel) -> dict:
    return {
        "id": e.id,
        "experiment_id": e.experiment_id,
        "metrics": json.loads(e.metrics) if isinstance(e.metrics, str) else e.metrics,
        "confusion_matrix": json.loads(e.confusion_matrix) if e.confusion_matrix else None,
        "per_class_metrics": json.loads(e.per_class_metrics) if e.per_class_metrics else None,
        "best_checkpoint": e.best_checkpoint,
        "insights": json.loads(e.insights) if e.insights else None,
        "created_at": e.created_at,
    }
