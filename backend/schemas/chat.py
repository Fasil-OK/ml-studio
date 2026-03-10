import json
from datetime import datetime
from typing import Any
from pydantic import BaseModel, field_validator


class ChatMessageCreate(BaseModel):
    content: str
    context: dict[str, Any] | None = None


class ChatMessageResponse(BaseModel):
    id: int
    project_id: str
    role: str
    content: str
    context: dict[str, Any] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("context", mode="before")
    @classmethod
    def parse_json_dict(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v
