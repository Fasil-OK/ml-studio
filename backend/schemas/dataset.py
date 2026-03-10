import json
from datetime import datetime
from typing import Any
from pydantic import BaseModel, field_validator


class DatasetResponse(BaseModel):
    id: str
    project_id: str
    name: str
    total_images: int | None
    num_classes: int | None
    class_names: list[str] | None = None
    class_counts: dict[str, int] | None = None
    image_stats: dict[str, Any] | None = None
    annotation_format: str | None
    quality_issues: list[dict[str, Any]] | None = None
    split_info: dict[str, int] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("class_names", "quality_issues", mode="before")
    @classmethod
    def parse_json_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("class_counts", "image_stats", "split_info", mode="before")
    @classmethod
    def parse_json_dict(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v
