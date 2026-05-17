import tempfile

from backend.core.config import get_settings
from backend.utils.audio import preprocess_audio, run_ffmpeg_normalize, save_temp_audio


async def process_audio(audio_file) -> str:
    settings = get_settings()
    temp_path = await save_temp_audio(audio_file)
    output_path = tempfile.mktemp(suffix="_optimized.wav", dir=settings.temp_audio_dir)
    try:
        return run_ffmpeg_normalize(temp_path, output_path)
    except Exception:
        return preprocess_audio(temp_path)
