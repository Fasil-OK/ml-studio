import json
from datetime import datetime
from typing import Any
from pydantic import BaseModel, field_validator


class EvaluationResponse(BaseModel):
    id: str
    experiment_id: str
    metrics: dict[str, Any]
    confusion_matrix: list[list[int]] | None = None
    per_class_metrics: list[dict[str, Any]] | None = None
    best_checkpoint: str | None
    insights: list[str] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("metrics", mode="before")
    @classmethod
    def parse_json_dict(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("confusion_matrix", "per_class_metrics", "insights", mode="before")
    @classmethod
    def parse_json_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v
