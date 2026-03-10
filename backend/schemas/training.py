import json
from datetime import datetime
from typing import Any
from pydantic import BaseModel, field_validator


class TrainingMetricResponse(BaseModel):
    epoch: int
    train_loss: float | None
    train_accuracy: float | None
    val_loss: float | None
    val_accuracy: float | None
    learning_rate: float | None
    epoch_duration: float | None
    gpu_memory_used: float | None
    extra_metrics: dict[str, Any] | None = None
    timestamp: datetime

    model_config = {"from_attributes": True}

    @field_validator("extra_metrics", mode="before")
    @classmethod
    def parse_json_dict(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v


class TrainingStatusResponse(BaseModel):
    experiment_id: str
    status: str
    current_epoch: int | None = None
    total_epochs: int | None = None
    metrics: list[TrainingMetricResponse] = []
