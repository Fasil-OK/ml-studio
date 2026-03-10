import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ws.manager import ws_manager

router = APIRouter()


@router.websocket("/ws/training/{experiment_id}")
async def training_websocket(websocket: WebSocket, experiment_id: str):
    await ws_manager.connect(f"training:{experiment_id}", websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "stop_training":
                from services.training_service import TrainingService
                TrainingService().stop(experiment_id)
            elif msg.get("type") == "ping":
                await ws_manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(f"training:{experiment_id}", websocket)
