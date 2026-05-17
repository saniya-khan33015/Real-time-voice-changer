import shutil
import sys
from pathlib import Path

from loguru import logger

from backend.core.config import settings
from backend.services.model_manager import list_models


REQUIRED_DIRS = (
    settings.cloned_voices_dir,
    settings.tts_models_dir,
    settings.xtts_models_dir,
    settings.temp_audio_dir,
    settings.output_audio_dir,
    settings.projects_dir,
    settings.logs_dir,
)


def ensure_directories() -> None:
    for directory in REQUIRED_DIRS:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_ffmpeg() -> dict:
    ffmpeg = shutil.which(settings.ffmpeg_path)
    if not ffmpeg:
        logger.warning("FFmpeg not found. MP3 export and advanced preprocessing may be unavailable.")
        return {"available": False, "path": None}
    logger.info("FFmpeg found: {}", ffmpeg)
    return {"available": True, "path": ffmpeg}


def check_torch() -> dict:
    try:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "cpu"
        logger.info("Torch available. CUDA={}, device={}", cuda, device)
        return {"available": True, "cuda": cuda, "device": device}
    except Exception as exc:
        logger.warning("Torch check failed: {}", exc)
        return {"available": False, "cuda": False, "device": "cpu", "error": str(exc)}


def check_tts() -> dict:
    model_roots = (settings.xtts_models_dir, settings.tts_models_dir)
    model_dirs = []
    for root in model_roots:
        if root.exists():
            for folder in sorted(path for path in root.rglob("*") if path.is_dir()):
                has_config = (folder / "config.json").exists()
                has_checkpoint = any(path.suffix.lower() in {".pth", ".pt"} for path in folder.glob("*"))
                if has_config and has_checkpoint:
                    model_dirs.append(str(folder))
    try:
        import TTS  # noqa: F401

        package_available = True
    except Exception:
        package_available = False
    backend = "coqui_xtts" if package_available and model_dirs else "dummy"
    if backend == "dummy":
        logger.warning("XTTS/Coqui model not fully available. Add config.json plus checkpoint under ai_models/xtts. Dummy local TTS fallback is active.")
    else:
        logger.info("XTTS loaded from local model folders")
    return {
        "available": backend != "dummy",
        "backend": backend,
        "package_available": package_available,
        "model_dirs": model_dirs,
    }


def check_models() -> dict:
    models = list_models()
    cloned = models.get("cloned", [])
    xtts = models.get("xtts", [])
    ready_xtts = [item["name"] for item in xtts if item.get("ready")]
    logger.info("XTTS models detected: {}", ", ".join(ready_xtts) if ready_xtts else "none")
    logger.info("Cloned voices detected: {}", len(cloned))
    return {"xtts_count": len(xtts), "ready_xtts": ready_xtts, "cloned_voice_count": len(cloned)}


def run_startup_checks() -> dict:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.logs_dir / "voice_studio.log",
        rotation="2 MB",
        retention=5,
        level=settings.log_level,
        backtrace=True,
        diagnose=True,
    )
    logger.info("Running Local AI Voice Studio startup checks")
    ensure_directories()
    status = {
        "ffmpeg": check_ffmpeg(),
        "torch": check_torch(),
        "xtts": check_tts(),
        "models": check_models(),
        "directories": [str(path) for path in REQUIRED_DIRS],
    }
    logger.info("API routes loaded")
    logger.info("Frontend connection ready on http://localhost:{}", settings.gradio_port)
    logger.info("Startup checks complete")
    return status
