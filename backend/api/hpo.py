from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.experiment import Experiment
from services.hpo_service import HPOService

router = APIRouter()
hpo_service = HPOService()


class HPORequest(BaseModel):
    n_trials: int = 20


@router.post("/experiments/{experiment_id}/hpo")
async def start_hpo(experiment_id: str, data: HPORequest, db: AsyncSession = Depends(get_db)):
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found")
    result = await hpo_service.start_hpo(experiment, data.n_trials, db)
    return result


@router.post("/experiments/{experiment_id}/hpo/stop")
async def stop_hpo(experiment_id: str):
    hpo_service.stop_hpo(experiment_id)
    return {"status": "stopped"}
