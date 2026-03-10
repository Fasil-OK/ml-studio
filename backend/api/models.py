from fastapi import APIRouter, Query

from services.model_service import ModelService

router = APIRouter()


@router.get("")
async def list_architectures(task_type: str = Query(..., description="classification|detection|segmentation")):
    service = ModelService()
    return service.list_architectures(task_type)


@router.get("/recommend")
async def recommend_architecture(
    task_type: str = Query(...),
    num_classes: int = Query(10),
    dataset_size: int = Query(1000),
    gpu_vram_mb: int = Query(4096),
):
    service = ModelService()
    return service.recommend(task_type, num_classes, dataset_size, gpu_vram_mb)
