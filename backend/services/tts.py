import json
import re
import tempfile
from pathlib import Path

from loguru import logger

from backend.core.config import get_settings
from backend.models.tts_wrapper import LocalTTSModel
from backend.services.model_manager import registry
from backend.utils.audio import concatenate_wavs, safe_name


def get_tts_model() -> LocalTTSModel:
    return registry.cached("tts:default", LocalTTSModel)


def profile_reference(profile_name: str | None) -> str | None:
    if not profile_name:
        return None
    metadata_path = get_settings().cloned_voices_dir / safe_name(profile_name) / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("sample_path")
    return None


def chunk_text(text: str, max_chars: int = 240) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts or [text.strip()]:
        if len(current) + len(part) + 1 <= max_chars:
            current = f"{current} {part}".strip()
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks or [" "]


async def generate_tts(text: str, speaker_name: str | None = None) -> str:
    return generate_speech_sync(text, speaker_name)


def generate_speech_sync(text: str, speaker_name: str | None = None) -> str:
    settings = get_settings()
    settings.output_audio_dir.mkdir(parents=True, exist_ok=True)
    model = get_tts_model()
    chunks = chunk_text(text)
    paths = []
    speaker_wav = profile_reference(speaker_name)
    logger.info("TTS generation requested: chunks={}, speaker={}", len(chunks), speaker_name or "default")
    for chunk in chunks:
        out = tempfile.mktemp(suffix="_tts_part.wav", dir=settings.temp_audio_dir)
        model.synthesize(chunk, speaker_name, out, speaker_wav=speaker_wav)
        paths.append(out)
    final = tempfile.mktemp(suffix="_tts.wav", dir=settings.output_audio_dir)
    concatenate_wavs(paths, final, pause_seconds=0.18)
    return final
