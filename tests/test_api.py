from fastapi.testclient import TestClient
from backend.app import app

def test_model_list():
    client = TestClient(app)
    resp = client.get("/api/model/list")
    assert resp.status_code == 200
    assert "models" in resp.json()

def test_tts_generate_fail():
    client = TestClient(app)
    resp = client.post("/api/tts/generate", data={"text": "hello", "speaker_name": "test"})
    assert resp.status_code == 200
    assert "audio_path" in resp.json()
