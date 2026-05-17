import json
from pathlib import Path
import re
from typing import Optional

import numpy as np
from loguru import logger

from backend.core.config import get_settings
from backend.utils.audio import normalize_audio
from backend.utils.audio_io import write_audio

try:
    from TTS.api import TTS
except Exception:
    TTS = None


class LocalTTSModel:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.backend = "dummy"
        self.model = None
        self.sample_rate = self.settings.audio_sample_rate
        self._load()

    def _find_coqui_model(self) -> Optional[Path]:
        for root in (self.settings.xtts_models_dir, self.settings.tts_models_dir):
            for marker in ("config.json", "model.pth", "model_file.pth"):
                matches = list(root.rglob(marker))
                if matches:
                    return matches[0].parent
        return None

    def _is_xtts_config(self, config: Path) -> bool:
        try:
            with config.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return str(data.get("model", "")).lower() == "xtts"
        except Exception:
            return False

    def _speaker_names(self) -> list[str]:
        if self.model is None:
            return []
        speakers = getattr(self.model, "speakers", None) or []
        if speakers:
            return list(speakers)
        manager = getattr(getattr(self.model.synthesizer, "tts_model", None), "speaker_manager", None)
        speaker_map = getattr(manager, "speakers", None) or {}
        return list(speaker_map.keys())

    def _load(self) -> None:
        if self.settings.tts_backend == "dummy" or TTS is None:
            logger.warning("Using local dummy TTS. Install Coqui TTS models locally for natural speech.")
            return
        model_dir = self._find_coqui_model()
        if not model_dir:
            logger.warning("No local Coqui model folder found in {}. Using dummy TTS.", self.settings.tts_models_dir)
            return
        try:
            config = model_dir / "config.json"
            checkpoint = next(iter(model_dir.glob("*.pth")), None)
            if checkpoint and config.exists():
                model_path = model_dir if self._is_xtts_config(config) else checkpoint
                self.model = TTS(model_path=str(model_path), config_path=str(config), progress_bar=False, gpu=False)
                self.backend = "coqui"
                self.sample_rate = getattr(self.model.synthesizer, "output_sample_rate", self.sample_rate)
                logger.info("Loaded local Coqui TTS model from {}", model_dir)
        except Exception as exc:
            logger.error("Failed to load local Coqui model: {}", exc)

    def _prepare_dialogue_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip())
        text = re.sub(r"^(speaker\s*\d+|speaker\s*[a-z]|narrator)\s*:\s*", "", text, flags=re.IGNORECASE)
        replacements = {
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "…": "...",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        contractions = {
            r"\bit's\b": "it is",
            r"\bthat's\b": "that is",
            r"\bwhat's\b": "what is",
            r"\byou're\b": "you are",
            r"\bI'm\b": "I am",
            r"\bI've\b": "I have",
            r"\bdon't\b": "do not",
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if text and text[-1] not in ".!?।":
            text += "."
        return text

    def _prepare_hindi_text(self, text: str) -> str:
        text = self._prepare_dialogue_text(text)
        technical_terms = {
            "टीटीएस": "TTS",
            "टी टी एस": "TTS",
            "एक्सटीटीएस": "XTTS",
            "वॉइस": "voice",
            "वॉयस": "voice",
            "क्लोनिंग": "cloning",
            "कन्वर्जन": "conversion",
        }
        for source, target in technical_terms.items():
            text = text.replace(source, target)
        replacements = {
            ",": ", ",
            ";": ", ",
            ":": ", ",
            "?": "? ",
            "!": "! ",
            ".": ". ",
            "|": "। ",
            "॥": "। ",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_language(self, language: Optional[str]) -> str:
        value = (language or "en").strip().lower()
        if value in {"hindi", "hi", "hin"}:
            return "hi"
        return "en"

    def _quality_kwargs(self, language: str, speed: float) -> dict:
        base = {
            "split_sentences": False,
            "do_sample": False,
            "temperature": 0.3,
            "top_p": 0.65,
            "top_k": 25,
            "repetition_penalty": 2.0,
            "length_penalty": 1.0,
            "speed": speed,
        }
        if language == "hi":
            return base
        return base

    def synthesize(
        self,
        text: str,
        speaker_name: Optional[str],
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
    ) -> str:
        language = self._normalize_language(language)
        text = self._prepare_hindi_text(text) if language == "hi" else self._prepare_dialogue_text(text)
        speed = min(1.5, max(0.75, float(speed or 1.0)))
        if not text:
            text = " "
        if self.backend == "coqui" and self.model is not None:
            logger.info(
                "Generating XTTS speech: chars={}, speaker={}, language={}, speed={}",
                len(text),
                speaker_name or "reference",
                language,
                speed,
            )
            kwargs = self._quality_kwargs(language, speed)
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            elif speaker_name:
                speakers = self._speaker_names()
                if speaker_name in speakers:
                    kwargs["speaker"] = speaker_name
                elif speakers:
                    logger.warning("Unknown TTS speaker '{}'. Using default speaker '{}'.", speaker_name, speakers[0])
                    kwargs["speaker"] = speakers[0]
            if getattr(self.model, "is_multi_lingual", False):
                kwargs["language"] = language
            wav = self.model.tts(text=text, **kwargs)
            write_audio(output_path, normalize_audio(np.asarray(wav, dtype=np.float32)), self.sample_rate)
            return output_path
        if speaker_wav and self.settings.xtts_require_reference_voice:
            raise RuntimeError(
                "XTTS reference voice mode is enabled, but the Coqui XTTS model is not loaded. "
                "Check ai_models/xtts/xtts_v2 and restart the backend."
            )
        logger.debug("Generating dummy TTS preview: chars={}, speaker={}", len(text), speaker_name or "default")
        return self._dummy_speech(text, speaker_name, output_path)

    def _dummy_speech(self, text: str, speaker_name: Optional[str], output_path: str) -> str:
        sr = self.sample_rate
        words = max(1, len(text.split()))
        duration = min(12.0, max(0.8, words * 0.22))
        identity = (speaker_name or "voice") + text[:16]
        base = 140 + (sum(ord(ch) for ch in identity) % 220)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        carrier = np.sin(2 * np.pi * base * t)
        harmonic = 0.35 * np.sin(2 * np.pi * base * 2.01 * t)
        prosody = 0.65 + 0.25 * np.sin(2 * np.pi * 2.2 * t)
        envelope = np.minimum(1.0, np.linspace(0, 10, t.size)) * np.minimum(1.0, np.linspace(10, 0, t.size))
        audio = 0.16 * (carrier + harmonic) * prosody * envelope
        write_audio(output_path, normalize_audio(audio.astype(np.float32), peak=0.8), sr)
        return output_path

    def unload(self) -> None:
        self.model = None
