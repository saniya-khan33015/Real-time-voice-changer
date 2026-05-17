from fastapi import WebSocket

async def handle_streaming(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            await websocket.send_bytes(data)
    except Exception:
        await websocket.close()
