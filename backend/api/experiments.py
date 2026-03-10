import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.project import Project
from models.dataset import Dataset
from models.experiment import Experiment
from schemas.experiment import ExperimentCreate, ExperimentResponse

router = APIRouter()


@router.post("/projects/{project_id}/experiments", response_model=ExperimentResponse)
async def create_experiment(project_id: str, data: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    dataset = await db.get(Dataset, data.dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset not found for this project")

    experiment = Experiment(
        project_id=project_id,
        dataset_id=data.dataset_id,
        architecture=data.architecture,
        pretrained=data.pretrained,
        hyperparameters=json.dumps(data.hyperparameters),
        resource_config=json.dumps(data.resource_config) if data.resource_config else None,
    )
    db.add(experiment)
    project.status = "configured"
    await db.commit()
    await db.refresh(experiment)
    return _to_response(experiment)


@router.get("/projects/{project_id}/experiments", response_model=list[ExperimentResponse])
async def list_experiments(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .order_by(Experiment.created_at)
    )
    return [_to_response(e) for e in result.scalars().all()]


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found")
    return _to_response(experiment)


def _to_response(e: Experiment) -> dict:
    return {
        "id": e.id,
        "project_id": e.project_id,
        "dataset_id": e.dataset_id,
        "architecture": e.architecture,
        "pretrained": e.pretrained,
        "hyperparameters": json.loads(e.hyperparameters),
        "resource_config": json.loads(e.resource_config) if e.resource_config else None,
        "status": e.status,
        "created_at": e.created_at,
    }
