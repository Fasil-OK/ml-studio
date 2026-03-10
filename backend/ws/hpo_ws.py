import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ws.manager import ws_manager

router = APIRouter()


@router.websocket("/ws/hpo/{experiment_id}")
async def hpo_websocket(websocket: WebSocket, experiment_id: str):
    await ws_manager.connect(f"hpo:{experiment_id}", websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws_manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(f"hpo:{experiment_id}", websocket)
