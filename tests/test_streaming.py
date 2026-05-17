from fastapi.testclient import TestClient
from backend.app import app

def test_streaming_endpoint_removed_for_xtts_only_app():
    client = TestClient(app)
    response = client.get("/ws/stream")
    assert response.status_code == 404
