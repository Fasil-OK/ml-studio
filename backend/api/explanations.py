from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.experiment import Experiment
from services.explanation_service import ExplanationService

router = APIRouter()


@router.post("/experiments/{experiment_id}/explain")
async def generate_explanation(
    experiment_id: str,
    image: UploadFile = File(...),
    method: str = Form("gradcam"),
    db: AsyncSession = Depends(get_db),
):
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found")

    service = ExplanationService()
    result = await service.explain(experiment, image, method, db)
    return result


@router.get("/experiments/{experiment_id}/explanations")
async def list_explanations(experiment_id: str):
    service = ExplanationService()
    return service.list_explanations(experiment_id)
