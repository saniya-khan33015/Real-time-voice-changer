from pathlib import Path
import os
import sys

os.environ.setdefault("XTTS_REQUIRE_REFERENCE_VOICE", "false")

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import app
from backend.utils.audio import tone_preview


def test_generate_voice(tmp_path):
    client = TestClient(app)
    speaker_wav = tmp_path / "speaker.wav"
    tone_preview("speaker sample", str(speaker_wav))
    with speaker_wav.open("rb") as handle:
        response = client.post(
            "/api/xtts/generate",
            data={"text": "Hello, this is a test.", "style": "original"},
            files={"speaker_wav": ("speaker.wav", handle, "audio/wav")},
        )
    assert response.status_code == 200
    result = response.json()
    assert "audio_path" in result
    assert Path(result["audio_path"]).exists()


if __name__ == "__main__":
    test_generate_voice()
