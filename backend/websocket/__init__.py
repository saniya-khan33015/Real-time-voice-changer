from fastapi import APIRouter, WebSocket
from backend.websocket.streaming import handle_streaming

router = APIRouter()

@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await handle_streaming(websocket)
