import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.core.config import get_settings
from backend.utils.audio import load_audio, preprocess_audio, safe_name, save_temp_audio, tone_preview
from backend.utils.audio_io import write_audio

try:
    import librosa
except Exception:
    librosa = None


def _embedding_from_audio(path: str) -> list[float]:
    audio, sr = load_audio(path, 16000)
    if audio.size == 0:
        return [0.0] * 32
    if librosa is not None:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=16)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rms = librosa.feature.rms(y=audio)
        vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), centroid.mean(axis=1), rms.mean(axis=1)])
    else:
        chunks = np.array_split(audio, 32)
        vector = np.asarray([float(np.sqrt(np.mean(chunk * chunk))) for chunk in chunks], dtype=np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    return vector.astype(float).tolist()


async def clone_voice(audio_file, speaker_name: str) -> dict:
    settings = get_settings()
    name = safe_name(speaker_name)
    profile_dir = settings.cloned_voices_dir / name
    samples_dir = profile_dir / "samples"
    profile_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    raw_path = await save_temp_audio(audio_file)
    processed_path = preprocess_audio(raw_path, settings.audio_sample_rate)
    sample_path = samples_dir / "reference.wav"
    shutil.copyfile(processed_path, sample_path)
    embedding = _embedding_from_audio(str(sample_path))
    embedding_path = profile_dir / "embedding.json"
    with embedding_path.open("w", encoding="utf-8") as handle:
        json.dump({"embedding": embedding}, handle, indent=2)

    preview_path = profile_dir / "preview.wav"
    try:
        audio, sr = load_audio(str(sample_path), settings.audio_sample_rate)
        audio = audio[: settings.audio_sample_rate * 6]
        write_audio(preview_path, audio, sr)
    except Exception:
        tone_preview(name, str(preview_path))

    metadata = {
        "name": name,
        "display_name": speaker_name,
        "backend": "coqui_xtts" if settings.tts_backend == "coqui" else "dummy_profile",
        "sample_path": str(sample_path),
        "embedding_path": str(embedding_path),
        "preview_path": str(preview_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (profile_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return {"status": "created", "profile": metadata}
