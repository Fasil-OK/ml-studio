from datetime import datetime
from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str
    task_type: str  # classification|detection|segmentation
    description: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    task_type: str
    description: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
