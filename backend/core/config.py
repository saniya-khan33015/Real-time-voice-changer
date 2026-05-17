from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except Exception:
    from pydantic import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    app_name: str = Field("XTTS Voice Cloning", env="APP_NAME")
    app_env: str = Field("production", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    gradio_port: int = Field(7860, env="GRADIO_PORT")

    root_dir: Path = Field(default_factory=lambda: Path.cwd())
    audio_sample_rate: int = Field(22050, env="AUDIO_SAMPLE_RATE")
    stream_sample_rate: int = Field(16000, env="STREAM_SAMPLE_RATE")
    audio_max_duration_seconds: int = Field(600, env="AUDIO_MAX_DURATION_SECONDS")
    model_cache_size: int = Field(2, env="MODEL_CACHE_SIZE")
    use_fp16: bool = Field(True, env="USE_FP16")
    use_onnx: bool = Field(True, env="USE_ONNX")
    quantize: bool = Field(False, env="QUANTIZE")
    tts_backend: Literal["coqui", "fish_speech", "openvoice", "dummy"] = Field("coqui", env="TTS_BACKEND")
    xtts_reference_max_seconds: float = Field(14.0, env="XTTS_REFERENCE_MAX_SECONDS")
    xtts_require_reference_voice: bool = Field(True, env="XTTS_REQUIRE_REFERENCE_VOICE")
    default_speaker_1_reference_audio: str = Field("", env="DEFAULT_SPEAKER_1_REFERENCE_AUDIO")
    default_speaker_2_reference_audio: str = Field("", env="DEFAULT_SPEAKER_2_REFERENCE_AUDIO")
    ffmpeg_path: str = Field("ffmpeg", env="FFMPEG_PATH")
    fish_speech_url: str = Field("http://127.0.0.1:8080/v1/tts", env="FISH_SPEECH_URL")
    fish_speech_api_key: str = Field("", env="FISH_SPEECH_API_KEY")
    fish_speech_temperature: float = Field(0.25, env="FISH_SPEECH_TEMPERATURE")
    fish_speech_top_p: float = Field(0.55, env="FISH_SPEECH_TOP_P")
    fish_speech_repetition_penalty: float = Field(1.25, env="FISH_SPEECH_REPETITION_PENALTY")
    fish_speech_chunk_length: int = Field(300, env="FISH_SPEECH_CHUNK_LENGTH")
    fish_speech_seed: int = Field(42, env="FISH_SPEECH_SEED")
    fish_speech_fallback_to_xtts: bool = Field(True, env="FISH_SPEECH_FALLBACK_TO_XTTS")
    fish_speech_speaker_1_reference_text: str = Field(
        "मोदी एक शब्द नहीं बोल रहे",
        env="FISH_SPEECH_SPEAKER_1_REFERENCE_TEXT",
    )
    fish_speech_speaker_2_reference_text: str = Field("", env="FISH_SPEECH_SPEAKER_2_REFERENCE_TEXT")
    @property
    def audio_dir(self) -> Path:
        return self.root_dir / "audio"

    @property
    def temp_audio_dir(self) -> Path:
        return self.audio_dir / "temp"

    @property
    def output_audio_dir(self) -> Path:
        return self.audio_dir / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "ai_models"

    @property
    def cloned_voices_dir(self) -> Path:
        return self.models_dir / "cloned_voices"

    @property
    def tts_models_dir(self) -> Path:
        return self.models_dir / "tts"

    @property
    def xtts_models_dir(self) -> Path:
        return self.models_dir / "xtts"

    @property
    def projects_dir(self) -> Path:
        return self.root_dir / "projects"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
