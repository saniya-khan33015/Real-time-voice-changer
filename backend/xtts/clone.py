import tempfile

from backend.core.config import get_settings
from backend.models.tts_wrapper import LocalTTSModel
from backend.services.model_manager import registry


def clone_voice(text, speaker_wav, output_path=None, language="en", speed=1.0):
    """
    Generate speech using the configured local TTS backend.
    """
    settings = get_settings()
    settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
    settings.output_audio_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = tempfile.mktemp(suffix="_xtts.wav", dir=settings.temp_audio_dir)
    model = registry.cached("tts:default", LocalTTSModel)
    return model.synthesize(text, None, output_path, speaker_wav=speaker_wav, language=language, speed=speed)
