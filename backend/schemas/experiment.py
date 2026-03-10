import json
from datetime import datetime
from typing import Any
from pydantic import BaseModel, field_validator


class ExperimentCreate(BaseModel):
    dataset_id: str
    architecture: str
    pretrained: bool = True
    hyperparameters: dict[str, Any]
    resource_config: dict[str, Any] | None = None


class ExperimentResponse(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    architecture: str
    pretrained: bool
    hyperparameters: dict[str, Any]
    resource_config: dict[str, Any] | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("hyperparameters", "resource_config", mode="before")
    @classmethod
    def parse_json_dict(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v
