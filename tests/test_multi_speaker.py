from pathlib import Path
import os
import sys

os.environ.setdefault("XTTS_REQUIRE_REFERENCE_VOICE", "false")

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import app
from backend.pipeline.xtts_pipeline import _chunk_dialogue_text, script_to_dialogues
from backend.utils.audio import tone_preview


def test_script_to_dialogues_maps_each_speaker_voice_and_style():
    dialogues = script_to_dialogues(
        "Speaker 1: Hello.\nSpeaker 2: Hi.\nSpeaker 1: Continue.",
        "voice_1.wav",
        "voice_2.wav",
        speaker_1_style="style_1",
        speaker_2_style="style_2",
        speaker_1_speed=1.0,
        speaker_2_speed=1.15,
    )
    assert [row["speaker_wav"] for row in dialogues] == ["voice_1.wav", "voice_2.wav", "voice_1.wav"]
    assert [row["style"] for row in dialogues] == ["style_1", "style_2", "style_1"]
    assert [row["speed"] for row in dialogues] == [1.0, 1.15, 1.0]


def test_chunk_dialogue_text_keeps_turns_short():
    chunks = _chunk_dialogue_text(
        "This is the first sentence. This is the second sentence. This is the third sentence.",
        max_chars=40,
    )
    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_script_to_dialogues_maps_three_speakers_and_language():
    dialogues = script_to_dialogues(
        "Speaker 1: Namaste.\nSpeaker 2: Main theek hoon.\nSpeaker 3: Aap dono se milkar accha laga.",
        "voice_1.wav",
        "voice_2.wav",
        voice_3_path="voice_3.wav",
        speaker_1_style="style_1",
        speaker_2_style="style_2",
        speaker_3_style="style_3",
        speaker_1_speed=1.0,
        speaker_2_speed=1.15,
        speaker_3_speed=1.2,
        language="Hindi",
    )
    assert [row["speaker_wav"] for row in dialogues] == ["voice_1.wav", "voice_2.wav", "voice_3.wav"]
    assert [row["style"] for row in dialogues] == ["style_1", "style_2", "style_3"]
    assert [row["speed"] for row in dialogues] == [1.0, 1.15, 1.2]
    assert [row["language"] for row in dialogues] == ["hi", "hi", "hi"]


def test_multi_speaker(tmp_path):
    client = TestClient(app)
    speaker_wav = tmp_path / "speaker.wav"
    tone_preview("speaker sample", str(speaker_wav))
    dialogues = [
        {"speaker": "Narrator", "style": "original", "text": "Welcome everyone.", "speaker_wav": str(speaker_wav)},
        {"speaker": "Speaker A", "style": "original", "text": "Hello there.", "speaker_wav": str(speaker_wav)},
    ]
    response = client.post("/api/xtts/multi-speaker", json=dialogues)
    assert response.status_code == 200
    result = response.json()
    assert "audio_path" in result
    assert Path(result["audio_path"]).exists()


def test_two_voice_conversation(tmp_path):
    client = TestClient(app)
    voice_1 = tmp_path / "voice_1.wav"
    voice_2 = tmp_path / "voice_2.wav"
    tone_preview("speaker one sample", str(voice_1))
    tone_preview("speaker two sample", str(voice_2))
    script = "Speaker 1: Welcome.\nSpeaker 2: Hello there."
    with voice_1.open("rb") as first, voice_2.open("rb") as second:
        response = client.post(
            "/api/xtts/conversation",
            data={
                "script": script,
                "speaker_1_style": "original",
                "speaker_2_style": "original",
                "speaker_1_speed": "1.0",
                "speaker_2_speed": "1.15",
            },
            files={
                "voice_1": ("voice_1.wav", first, "audio/wav"),
                "voice_2": ("voice_2.wav", second, "audio/wav"),
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert "audio_path" in result
    assert Path(result["audio_path"]).exists()


def test_two_voice_conversation_rejects_unknown_speaker_label(tmp_path):
    client = TestClient(app)
    voice_1 = tmp_path / "voice_1.wav"
    voice_2 = tmp_path / "voice_2.wav"
    tone_preview("speaker one sample", str(voice_1))
    tone_preview("speaker two sample", str(voice_2))
    script = "Speaker 1: Welcome.\nSpeaker 3: This should not be accepted."
    with voice_1.open("rb") as first, voice_2.open("rb") as second:
        response = client.post(
            "/api/xtts/conversation",
            data={"script": script},
            files={
                "voice_1": ("voice_1.wav", first, "audio/wav"),
                "voice_2": ("voice_2.wav", second, "audio/wav"),
            },
        )
    assert response.status_code == 500
    assert "Only Speaker 1 and Speaker 2 are allowed" in response.json()["detail"]


def test_three_voice_conversation_endpoint(tmp_path):
    client = TestClient(app)
    voice_1 = tmp_path / "voice_1.wav"
    voice_2 = tmp_path / "voice_2.wav"
    voice_3 = tmp_path / "voice_3.wav"
    tone_preview("speaker one sample", str(voice_1))
    tone_preview("speaker two sample", str(voice_2))
    tone_preview("speaker three sample", str(voice_3))
    script = "Speaker 1: Hello.\nSpeaker 2: Hi.\nSpeaker 3: Nice to meet you."
    with voice_1.open("rb") as first, voice_2.open("rb") as second, voice_3.open("rb") as third:
        response = client.post(
            "/api/xtts/conversation",
            data={
                "script": script,
                "speaker_1_style": "original",
                "speaker_2_style": "original",
                "speaker_3_style": "original",
                "speaker_1_speed": "1.0",
                "speaker_2_speed": "1.15",
                "speaker_3_speed": "1.15",
                "language": "en",
            },
            files={
                "voice_1": ("voice_1.wav", first, "audio/wav"),
                "voice_2": ("voice_2.wav", second, "audio/wav"),
                "voice_3": ("voice_3.wav", third, "audio/wav"),
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert "audio_path" in result
    assert Path(result["audio_path"]).exists()


if __name__ == "__main__":
    test_multi_speaker()
