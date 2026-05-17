import tempfile
from pathlib import Path

import requests

from backend.core.config import get_settings

try:
    import ormsgpack
except Exception:
    ormsgpack = None


def synthesize_fish_speech_local(
    text: str,
    speaker_wav: str,
    reference_text: str,
    output_path: str | None = None,
) -> str:
    settings = get_settings()
    if ormsgpack is None:
        raise RuntimeError("Fish Speech local backend requires ormsgpack. Run: pip install ormsgpack")
    if not Path(speaker_wav).exists():
        raise FileNotFoundError(f"Fish Speech reference audio not found: {speaker_wav}")
    if not reference_text.strip():
        raise RuntimeError(
            "Fish Speech local backend needs reference text for the speaker audio. "
            "Set FISH_SPEECH_SPEAKER_1_REFERENCE_TEXT or FISH_SPEECH_SPEAKER_2_REFERENCE_TEXT in .env."
        )

    settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = tempfile.mktemp(suffix="_fish_speech.wav", dir=settings.temp_audio_dir)

    with open(speaker_wav, "rb") as handle:
        reference_audio = handle.read()

    payload = {
        "text": text,
        "references": [{"audio": reference_audio, "text": reference_text}],
        "reference_id": None,
        "format": "wav",
        "latency": "normal",
        "max_new_tokens": 1024,
        "chunk_length": min(300, max(100, int(settings.fish_speech_chunk_length))),
        "top_p": min(1.0, max(0.0, float(settings.fish_speech_top_p))),
        "repetition_penalty": float(settings.fish_speech_repetition_penalty),
        "temperature": min(1.0, max(0.0, float(settings.fish_speech_temperature))),
        "streaming": False,
        "use_memory_cache": "on",
        "seed": int(settings.fish_speech_seed),
    }
    headers = {"content-type": "application/msgpack"}
    if settings.fish_speech_api_key.strip():
        headers["authorization"] = f"Bearer {settings.fish_speech_api_key.strip()}"

    response = requests.post(
        settings.fish_speech_url,
        params={"format": "msgpack"},
        data=ormsgpack.packb(payload),
        headers=headers,
        timeout=300,
    )
    if response.status_code != 200:
        detail = response.text[:500]
        raise RuntimeError(f"Fish Speech local TTS failed: {response.status_code} {detail}")

    with open(output_path, "wb") as handle:
        handle.write(response.content)
    return output_path
