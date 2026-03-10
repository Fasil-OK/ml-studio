import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.project import Project
from models.chat_message import ChatMessage
from schemas.chat import ChatMessageCreate, ChatMessageResponse

router = APIRouter()


@router.get("/projects/{project_id}/chat/history", response_model=list[ChatMessageResponse])
async def get_chat_history(project_id: str, limit: int = 50, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.project_id == project_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    messages = list(reversed(result.scalars().all()))
    return [_parse_message(m) for m in messages]


def _parse_message(m: ChatMessage) -> dict:
    return {
        "id": m.id,
        "project_id": m.project_id,
        "role": m.role,
        "content": m.content,
        "context": json.loads(m.context) if m.context else None,
        "created_at": m.created_at,
    }
