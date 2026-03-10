import json

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.project import Project
from models.dataset import Dataset
from schemas.dataset import DatasetResponse
from services.dataset_service import DatasetService
from config import settings

router = APIRouter()


@router.post("/projects/{project_id}/dataset", response_model=DatasetResponse)
async def upload_dataset(
    project_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    service = DatasetService()
    dataset = await service.process_upload(project_id, project.task_type, file, db)
    return _parse_dataset(dataset)


@router.get("/projects/{project_id}/dataset", response_model=DatasetResponse | None)
async def get_dataset(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    )
    dataset = result.scalars().first()
    if not dataset:
        return None
    # Parse JSON fields
    return _parse_dataset(dataset)


@router.get("/projects/{project_id}/dataset/samples")
async def get_samples(
    project_id: str,
    class_name: str | None = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    )
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(404, "No dataset found")

    service = DatasetService()
    samples = service.get_sample_images(dataset.path, class_name, limit)
    return {"samples": samples}


@router.post("/projects/{project_id}/dataset/analyze", response_model=DatasetResponse)
async def reanalyze_dataset(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    )
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(404, "No dataset found")

    project = await db.get(Project, project_id)
    service = DatasetService()
    updated = await service.analyze_dataset(dataset, project.task_type, db)
    return _parse_dataset(updated)


def _parse_dataset(dataset: Dataset) -> dict:
    return {
        "id": dataset.id,
        "project_id": dataset.project_id,
        "name": dataset.name,
        "total_images": dataset.total_images,
        "num_classes": dataset.num_classes,
        "class_names": json.loads(dataset.class_names) if dataset.class_names else None,
        "class_counts": json.loads(dataset.class_counts) if dataset.class_counts else None,
        "image_stats": json.loads(dataset.image_stats) if dataset.image_stats else None,
        "annotation_format": dataset.annotation_format,
        "quality_issues": json.loads(dataset.quality_issues) if dataset.quality_issues else None,
        "split_info": json.loads(dataset.split_info) if dataset.split_info else None,
        "created_at": dataset.created_at,
    }
