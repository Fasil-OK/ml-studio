from fastapi import APIRouter

from services.resource_service import ResourceService

router = APIRouter()
resource_service = ResourceService()


@router.get("")
async def get_resources():
    return resource_service.get_system_resources()


@router.get("/recommend")
async def recommend_resources(architecture: str = "", batch_size: int = 32):
    return resource_service.recommend(architecture, batch_size)
