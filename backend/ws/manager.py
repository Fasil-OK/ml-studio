import json
from collections import defaultdict

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, room: str, websocket: WebSocket):
        await websocket.accept()
        self._connections[room].append(websocket)

    def disconnect(self, room: str, websocket: WebSocket):
        self._connections[room].remove(websocket)
        if not self._connections[room]:
            del self._connections[room]

    async def broadcast(self, room: str, data: dict):
        message = json.dumps(data)
        dead = []
        for ws in self._connections.get(room, []):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections[room].remove(ws)

    async def send_personal(self, websocket: WebSocket, data: dict):
        await websocket.send_text(json.dumps(data))


ws_manager = ConnectionManager()
