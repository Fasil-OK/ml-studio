import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ws.manager import ws_manager
from services.chat_service import ChatService

router = APIRouter()


@router.websocket("/ws/chat/{project_id}")
async def chat_websocket(websocket: WebSocket, project_id: str):
    await ws_manager.connect(f"chat:{project_id}", websocket)
    chat_service = ChatService()
    cancel_event: asyncio.Event | None = None

    async def _stream_response(content: str, context: dict):
        nonlocal cancel_event
        cancel_event = asyncio.Event()
        try:
            async for chunk in chat_service.stream_response(
                project_id, content, context, cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    break
                await ws_manager.send_personal(websocket, {
                    "type": "chunk",
                    "content": chunk,
                })
        except Exception:
            pass
        finally:
            await ws_manager.send_personal(websocket, {"type": "end"})
            cancel_event = None

    stream_task: asyncio.Task | None = None

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "stop":
                if cancel_event:
                    cancel_event.set()
                if stream_task and not stream_task.done():
                    # Wait briefly for clean shutdown
                    try:
                        await asyncio.wait_for(stream_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        stream_task.cancel()

            elif msg.get("type") == "message":
                content = msg.get("content", "")
                context = msg.get("context", {})
                # Cancel any ongoing stream
                if cancel_event:
                    cancel_event.set()
                if stream_task and not stream_task.done():
                    try:
                        await asyncio.wait_for(stream_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        stream_task.cancel()
                # Start streaming in background task so we can receive stop
                stream_task = asyncio.create_task(_stream_response(content, context))

    except WebSocketDisconnect:
        if cancel_event:
            cancel_event.set()
        if stream_task and not stream_task.done():
            stream_task.cancel()
        ws_manager.disconnect(f"chat:{project_id}", websocket)
