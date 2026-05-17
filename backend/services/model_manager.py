import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from backend.core.config import get_settings
from backend.utils.audio import safe_name


MODEL_EXTENSIONS = {".pth", ".pt", ".onnx"}
TTS_MARKERS = {"config.json", "model.pth", "model_file.pth"}


class ModelRegistry:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def _metadata(self, folder: Path) -> dict:
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _discover_folder_models(self, root: Path) -> list[dict]:
        root.mkdir(parents=True, exist_ok=True)
        results = []
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            files = [path for path in folder.rglob("*") if path.suffix.lower() in MODEL_EXTENSIONS]
            metadata = self._metadata(folder)
            results.append(
                {
                    "name": folder.name,
                    "path": str(folder),
                    "files": [str(path) for path in files],
                    "preview": str(folder / "preview.wav") if (folder / "preview.wav").exists() else None,
                    "metadata": metadata,
                    "ready": bool(files) or metadata.get("backend") in {"coqui_xtts", "dummy_profile"},
                }
            )
        return results

    def list_cloned_voices(self) -> list[dict]:
        return self._discover_folder_models(self.settings.cloned_voices_dir)

    def list_tts_models(self) -> list[dict]:
        results = []
        for root in (self.settings.xtts_models_dir, self.settings.tts_models_dir):
            root.mkdir(parents=True, exist_ok=True)
            for folder in sorted(path for path in root.iterdir() if path.is_dir()):
                files = [path for path in folder.rglob("*") if path.name in TTS_MARKERS]
                results.append(
                    {
                        "name": folder.name,
                        "path": str(folder),
                        "files": [str(path) for path in files],
                        "metadata": self._metadata(folder),
                        "ready": bool(files),
                    }
                )
        return results

    def list_models(self) -> dict:
        return {"xtts": self.list_tts_models(), "cloned": self.list_cloned_voices()}

    def save_metadata(self, model_type: str, name: str, metadata: dict) -> dict:
        root = self.settings.cloned_voices_dir
        folder = root / safe_name(name)
        folder.mkdir(parents=True, exist_ok=True)
        metadata_path = folder / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return {"status": "saved", "metadata_path": str(metadata_path)}

    def delete_model(self, model_type: str, name: str) -> dict:
        root = self.settings.cloned_voices_dir
        folder = root / safe_name(name)
        if not folder.exists():
            return {"error": "model not found"}
        shutil.rmtree(folder)
        self.unload(f"{model_type}:{name}")
        return {"status": "deleted"}

    def cached(self, key: str, factory: Callable[[], Any]) -> Any:
        if key in self._cache:
            self._cache.move_to_end(key)
            logger.debug("Model cache hit: {}", key)
            return self._cache[key]
        logger.info("Loading model into cache: {}", key)
        model = factory()
        self._cache[key] = model
        while len(self._cache) > self.settings.model_cache_size:
            old_key, old = self._cache.popitem(last=False)
            logger.info("Unloading model from cache: {}", old_key)
            unload = getattr(old, "unload", None)
            if callable(unload):
                unload()
        return model

    def unload(self, key: str | None = None) -> None:
        if key:
            model = self._cache.pop(key, None)
            if model and hasattr(model, "unload"):
                model.unload()
            return
        for model in self._cache.values():
            if hasattr(model, "unload"):
                model.unload()
        self._cache.clear()


registry = ModelRegistry()


def list_models() -> dict:
    return registry.list_models()


def list_cloned_voices() -> list[dict]:
    return registry.list_cloned_voices()


def save_model_metadata(model_type: str, name: str, metadata: dict) -> dict:
    return registry.save_metadata(model_type, name, metadata)


def delete_model(model_type: str, name: str) -> dict:
    return registry.delete_model(model_type, name)


def switch_model(model_name: str) -> dict:
    return {
        "active_model": model_name,
        "detail": "Model switching is not used by the XTTS cloning flow.",
    }
