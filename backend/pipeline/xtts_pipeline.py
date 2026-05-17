import os
import re
import tempfile
import time
from pathlib import Path

from loguru import logger

from backend.core.config import get_settings
from backend.utils.audio import (
    concatenate_wavs,
    limit_dialogue_clip,
    preprocess_audio,
    preprocess_reference_audio,
    stabilize_hindi_voice,
)


def _normalize_audio_path(path: str) -> str:
    cleaned = str(path or "").strip().strip("\"'")
    if len(cleaned) >= 3 and cleaned[1] == ":":
        drive = cleaned[:2]
        rest = cleaned[2:].replace("/", "\\")
        while "\\\\" in rest:
            rest = rest.replace("\\\\", "\\")
        return drive + rest
    return cleaned


def _normalize_language(language: str | None) -> str:
    value = (language or "en").strip().lower()
    if value in {"hindi", "hi", "hin"}:
        return "hi"
    return "en"


def _speaker_index_from_label(label: str) -> int | None:
    match = re.search(r"\bspeaker\s*(\d+)\b", label or "", flags=re.IGNORECASE)
    if not match:
        return None
    index = int(match.group(1)) - 1
    return index if index >= 0 else None


def _is_strict_two_speaker_mode(voice_3_path: str | None) -> bool:
    return not voice_3_path


def _chunk_dialogue_text(text: str, max_chars: int = 180) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    for source, target in {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "…": "...",
    }.items():
        cleaned = cleaned.replace(source, target)
    if not cleaned:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?।])\s+", cleaned) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences or [cleaned]:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
        current = sentence
    if current:
        chunks.append(current)
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue
        words = chunk.split()
        current_words: list[str] = []
        for word in words:
            candidate = " ".join([*current_words, word]).strip()
            if current_words and len(candidate) > max_chars:
                final_chunks.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            final_chunks.append(" ".join(current_words))
    return final_chunks


def generate_voice(text, speaker_wav, style="original", output_path=None, language="en", speed=1.0):
    """Pure XTTS voice cloning from the uploaded reference voice."""
    start = time.time()
    settings = get_settings()
    settings.output_audio_dir.mkdir(parents=True, exist_ok=True)
    style = style or "original"
    language = _normalize_language(language)
    speed = min(1.5, max(0.75, float(speed or 1.0)))
    logger.info(
        "XTTS voice cloning started: language={}, speed={}, chars={}",
        language,
        speed,
        len(text or ""),
    )

    from backend.xtts.clone import clone_voice

    xtts_temp = clone_voice(text, speaker_wav, language=language, speed=speed)
    processed_path = preprocess_audio(xtts_temp, settings.audio_sample_rate)
    if output_path:
        target = Path(output_path)
        if not target.is_absolute():
            target = settings.output_audio_dir / target.name
    else:
        target = Path(tempfile.mktemp(suffix="_xtts_clone.wav", dir=settings.output_audio_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(processed_path, target)

    elapsed = time.time() - start
    logger.info("XTTS voice cloning complete: seconds={:.2f}", elapsed)
    return {
        "audio_path": str(target),
        "style": "xtts_clone",
        "language": language,
        "speed": speed,
        "processing_time": elapsed,
    }


def get_available_styles():
    return ["xtts_clone"]


def script_to_dialogues(
    script: str,
    voice_1_path: str,
    voice_2_path: str,
    voice_3_path: str | None = None,
    speaker_1_style: str = "original",
    speaker_2_style: str = "original",
    speaker_3_style: str = "original",
    speaker_1_speed: float = 1.0,
    speaker_2_speed: float = 1.0,
    speaker_3_speed: float = 1.0,
    speaker_2_backend: str = "local",
    language: str = "en",
) -> list[dict]:
    """Convert a script into turn rows using uploaded reference voices."""
    settings = get_settings()
    speaker_config: dict[str, dict[str, str]] = {}
    next_voice = 0
    speakers = [
        {
            "speaker_wav": voice_1_path,
            "style": speaker_1_style or "original",
            "speed": speaker_1_speed,
            "backend": "local",
            "reference_max_seconds": settings.xtts_reference_max_seconds,
            "reference_text": settings.fish_speech_speaker_1_reference_text,
        },
        {
            "speaker_wav": voice_2_path,
            "style": speaker_2_style or "original",
            "speed": speaker_2_speed,
            "backend": speaker_2_backend or "local",
            "reference_max_seconds": settings.xtts_reference_max_seconds,
            "reference_text": settings.fish_speech_speaker_2_reference_text,
        },
    ]
    if voice_3_path:
        speakers.append({
            "speaker_wav": voice_3_path,
            "style": speaker_3_style or "original",
            "speed": speaker_3_speed,
            "reference_max_seconds": settings.xtts_reference_max_seconds,
        })
    dialogues: list[dict] = []
    fallback_speaker = "Speaker 1"
    language = _normalize_language(language)
    strict_two_speaker_mode = _is_strict_two_speaker_mode(voice_3_path)

    for raw in (script or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker = speaker.strip() or fallback_speaker
            text = text.strip()
        else:
            if strict_two_speaker_mode:
                raise ValueError(f"Every dialogue line must start with Speaker 1: or Speaker 2:. Problem line: {line}")
            speaker = f"Speaker {(len(dialogues) % 2) + 1}"
            text = line
        if not text:
            continue
        if strict_two_speaker_mode:
            speaker_index = _speaker_index_from_label(speaker)
            if speaker_index not in {0, 1}:
                raise ValueError(f"Only Speaker 1 and Speaker 2 are allowed. Problem label: {speaker}")
            speaker_config[speaker] = speakers[speaker_index]
        if speaker not in speaker_config:
            speaker_index = _speaker_index_from_label(speaker)
            if speaker_index is not None:
                if speaker_index >= len(speakers):
                    raise ValueError(f"{speaker} was used in the script, but no voice sample was provided for it")
                speaker_config[speaker] = speakers[speaker_index]
            else:
                if next_voice >= len(speakers):
                    raise ValueError(f"No voice sample is available for script label: {speaker}")
                speaker_config[speaker] = speakers[next_voice]
                next_voice += 1
        dialogues.append({
            "speaker": speaker,
            "text": text,
            "speaker_wav": speaker_config[speaker]["speaker_wav"],
            "style": speaker_config[speaker]["style"],
            "speed": speaker_config[speaker].get("speed", 1.0),
            "backend": speaker_config[speaker].get("backend", "local"),
            "reference_max_seconds": speaker_config[speaker].get("reference_max_seconds", 30.0),
            "reference_text": speaker_config[speaker].get("reference_text", ""),
            "language": language,
        })
    return dialogues


def generate_conversation_from_script(
    script: str,
    voice_1_path: str,
    voice_2_path: str,
    voice_3_path: str | None = None,
    speaker_1_style: str = "original",
    speaker_2_style: str = "original",
    speaker_3_style: str = "original",
    speaker_1_speed: float = 1.0,
    speaker_2_speed: float = 1.0,
    speaker_3_speed: float = 1.0,
    speaker_2_backend: str = "local",
    language: str = "en",
    pause_seconds: float = 0.05,
) -> str:
    voice_1_path = _normalize_audio_path(voice_1_path)
    voice_2_path = _normalize_audio_path(voice_2_path)
    voice_3_path = _normalize_audio_path(voice_3_path) if voice_3_path else None
    voice_rows = [("Sample Voice 1", voice_1_path), ("Sample Voice 2", voice_2_path)]
    if voice_3_path:
        voice_rows.append(("Sample Voice 3", voice_3_path))
    for label, path in voice_rows:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    dialogues = script_to_dialogues(
        script,
        voice_1_path,
        voice_2_path,
        voice_3_path=voice_3_path,
        speaker_1_style=speaker_1_style,
        speaker_2_style=speaker_2_style,
        speaker_3_style=speaker_3_style,
        speaker_1_speed=speaker_1_speed,
        speaker_2_speed=speaker_2_speed,
        speaker_3_speed=speaker_3_speed,
        speaker_2_backend=speaker_2_backend,
        language=language,
    )
    if not dialogues:
        raise ValueError("Dialogue/script text is required")
    return generate_multi_speaker(dialogues, pause_seconds=pause_seconds, language=language)


def generate_two_voice_conversation(
    script: str,
    voice_1_path: str,
    voice_2_path: str,
    speaker_1_style: str = "original",
    speaker_2_style: str = "original",
    speaker_1_speed: float = 1.0,
    speaker_2_speed: float = 1.0,
    speaker_2_backend: str = "local",
    language: str = "en",
    pause_seconds: float = 0.05,
) -> str:
    return generate_conversation_from_script(
        script=script,
        voice_1_path=voice_1_path,
        voice_2_path=voice_2_path,
        speaker_1_style=speaker_1_style,
        speaker_2_style=speaker_2_style,
        speaker_1_speed=speaker_1_speed,
        speaker_2_speed=speaker_2_speed,
        speaker_2_backend=speaker_2_backend,
        language=language,
        pause_seconds=pause_seconds,
    )


def generate_multi_speaker(dialogues, output_path=None, pause_seconds=0.05, language="en"):
    """Generate each turn sequentially, then merge with stable pause insertion."""
    settings = get_settings()
    settings.output_audio_dir.mkdir(parents=True, exist_ok=True)
    conversation_files = []
    generated_files = []
    cleaned_references: dict[str, str] = {}
    original_references: set[str] = set()
    logger.info("Multi-speaker generation started: turns={}", len(dialogues or []))

    try:
        for index, dialogue in enumerate(dialogues or [], start=1):
            if not dialogue.get("text") or not dialogue.get("speaker_wav"):
                raise ValueError(f"Dialogue row {index} is missing text or speaker_wav")
            speaker_wav = _normalize_audio_path(dialogue["speaker_wav"])
            original_references.add(speaker_wav)
            if not Path(speaker_wav).exists():
                raise FileNotFoundError(f"Dialogue row {index} reference voice not found: {speaker_wav}")
            if speaker_wav not in cleaned_references:
                cleaned_references[speaker_wav] = preprocess_reference_audio(
                    speaker_wav,
                    settings.audio_sample_rate,
                    max_seconds=float(dialogue.get("reference_max_seconds", 30.0)),
                )
            line_language = dialogue.get("language", language)
            chunk_limit = 110 if _normalize_language(line_language) == "hi" else 130
            chunks = _chunk_dialogue_text(dialogue["text"], max_chars=chunk_limit)
            turn_segments = []
            for chunk in chunks:
                if settings.tts_backend == "fish_speech":
                    from backend.services.fish_speech_local import synthesize_fish_speech_local

                    try:
                        audio_path = synthesize_fish_speech_local(
                            chunk,
                            cleaned_references[speaker_wav],
                            dialogue.get("reference_text", ""),
                        )
                    except Exception:
                        if not settings.fish_speech_fallback_to_xtts:
                            raise
                        logger.exception("Fish Speech generation failed; falling back to XTTS for this chunk")
                        result = generate_voice(
                            chunk,
                            cleaned_references[speaker_wav],
                            dialogue.get("style", "original"),
                            language=line_language,
                            speed=dialogue.get("speed", 1.0),
                        )
                        audio_path = result["audio_path"]
                else:
                    result = generate_voice(
                        chunk,
                        cleaned_references[speaker_wav],
                        dialogue.get("style", "original"),
                        language=line_language,
                        speed=dialogue.get("speed", 1.0),
                    )
                    audio_path = result["audio_path"]
                limit_dialogue_clip(
                    audio_path,
                    chunk,
                    settings.audio_sample_rate,
                    language=line_language,
                )
                if _normalize_language(line_language) == "hi":
                    stabilize_hindi_voice(audio_path, settings.audio_sample_rate)
                generated_files.append(audio_path)
                turn_segments.append(audio_path)
            if len(turn_segments) > 1:
                turn_path = tempfile.mktemp(suffix="_turn.wav", dir=settings.temp_audio_dir)
                concatenate_wavs(turn_segments, turn_path, pause_seconds=0.03)
                generated_files.append(turn_path)
                conversation_files.append(turn_path)
            else:
                conversation_files.extend(turn_segments)

        if output_path is None:
            output_path = tempfile.mktemp(suffix="_conversation.wav", dir=settings.output_audio_dir)
        concatenate_wavs(conversation_files, output_path, pause_seconds=pause_seconds)
    finally:
        for path in generated_files:
            if os.path.exists(path):
                os.remove(path)
        for path in cleaned_references.values():
            if path not in original_references and os.path.exists(path):
                os.remove(path)

    logger.info("Multi-speaker generation complete: {}", output_path)
    return output_path
