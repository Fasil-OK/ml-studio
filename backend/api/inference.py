from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.experiment import Experiment
from services.inference_service import InferenceService

router = APIRouter()
inference_service = InferenceService()


@router.post("/experiments/{experiment_id}/predict")
async def predict(
    experiment_id: str,
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found")

    result = await inference_service.predict(experiment, image)
    return result
