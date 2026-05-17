import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from loguru import logger

from backend.core.config import get_settings
from backend.utils.audio_io import read_audio, write_audio

try:
    import librosa
except Exception:
    librosa = None


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return cleaned.strip("_") or "untitled"


async def save_temp_audio(audio_file) -> str:
    settings = get_settings()
    settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(getattr(audio_file, "filename", "") or "audio.wav").suffix or ".wav"
    fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=settings.temp_audio_dir)
    with open(fd, "wb") as handle:
        handle.write(await audio_file.read())
    return temp_path


def load_audio(path: str, sample_rate: Optional[int] = None) -> tuple[np.ndarray, int]:
    target_sr = sample_rate or get_settings().audio_sample_rate
    audio, sr = _load_mono(path, target_sr)
    return normalize_audio(audio), sr


def _load_mono(path: str, sample_rate: int) -> tuple[np.ndarray, int]:
    if librosa is not None:
        return librosa.load(path, sr=sample_rate, mono=True)
    from scipy import signal

    audio, sr = read_audio(path)
    audio = audio.mean(axis=1) if getattr(audio, "ndim", 1) > 1 else audio
    if sr != sample_rate and audio.size:
        gcd = math.gcd(sr, sample_rate)
        audio = signal.resample_poly(audio, sample_rate // gcd, sr // gcd)
        sr = sample_rate
    return np.asarray(audio, dtype=np.float32), sr


def extract_audio_track(input_path: str, sample_rate: Optional[int] = None) -> str:
    settings = get_settings()
    settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
    sr = sample_rate or settings.audio_sample_rate
    output_path = tempfile.mktemp(suffix="_track.wav", dir=settings.temp_audio_dir)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required to extract audio from video reference files")
    subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-af",
            "highpass=f=70,lowpass=f=7600,loudnorm=I=-18:TP=-2:LRA=9",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def clean_reference_with_ffmpeg(input_path: str, sample_rate: Optional[int] = None) -> str:
    settings = get_settings()
    settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
    sr = sample_rate or settings.audio_sample_rate
    output_path = tempfile.mktemp(suffix="_clean_ref.wav", dir=settings.temp_audio_dir)
    if not ffmpeg_available():
        return input_path
    subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-af",
            "highpass=f=85,lowpass=f=7200,afftdn=nf=-28,acompressor=threshold=-22dB:ratio=2.2:attack=8:release=120,loudnorm=I=-19:TP=-2:LRA=8",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def transcode_reference_with_ffmpeg(input_path: str, sample_rate: Optional[int] = None) -> str:
    settings = get_settings()
    settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
    sr = sample_rate or settings.audio_sample_rate
    output_path = tempfile.mktemp(suffix="_decoded_ref.wav", dir=settings.temp_audio_dir)
    if not ffmpeg_available():
        return input_path
    subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def _select_loud_reference_window(audio: np.ndarray, sample_rate: int, max_seconds: float) -> np.ndarray:
    max_samples = int(sample_rate * max_seconds)
    if audio.size <= max_samples or max_samples <= 0:
        return audio
    frame = max(1, int(sample_rate * 0.25))
    hop = max(1, frame // 2)
    if audio.size <= frame:
        return audio[:max_samples]
    scores: list[float] = []
    offsets: list[int] = []
    for start in range(0, audio.size - frame + 1, hop):
        chunk = audio[start:start + frame]
        scores.append(float(np.sqrt(np.mean(chunk * chunk))))
        offsets.append(start)
    if not scores:
        return audio[:max_samples]
    keep_frames = max(1, int(max_samples / hop))
    best_score = -1.0
    best_offset = 0
    for index in range(0, max(1, len(scores) - keep_frames + 1)):
        score = float(np.mean(scores[index:index + keep_frames]))
        if score > best_score:
            best_score = score
            best_offset = offsets[index]
    return audio[best_offset:best_offset + max_samples]


def is_clean_xtts_reference(input_path: str, sample_rate: Optional[int] = None, max_seconds: float = 30.0) -> bool:
    if Path(input_path).suffix.lower() != ".wav":
        return False
    sr = sample_rate or get_settings().audio_sample_rate
    try:
        audio, file_sr = read_audio(input_path)
    except Exception:
        return False
    channels = 1 if getattr(audio, "ndim", 1) == 1 else audio.shape[1]
    duration = len(audio) / float(file_sr or sr)
    return file_sr == sr and channels == 1 and 1.5 <= duration <= max_seconds


def preprocess_audio(input_path: str, sample_rate: Optional[int] = None, normalize: bool = True) -> str:
    settings = get_settings()
    settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
    sr = sample_rate or settings.audio_sample_rate
    audio, _ = _load_mono(input_path, sr)
    if audio.size and librosa is not None:
        audio = librosa.effects.trim(audio, top_db=35)[0]
    if normalize:
        audio = normalize_audio(audio)
    output_path = tempfile.mktemp(suffix="_pre.wav", dir=settings.temp_audio_dir)
    write_audio(output_path, audio, sr)
    return output_path


def preprocess_reference_audio(
    input_path: str,
    sample_rate: Optional[int] = None,
    max_seconds: float = 22.0,
    preserve_identity: bool = True,
) -> str:
    settings = get_settings()
    settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
    sr = sample_rate or settings.audio_sample_rate
    if preserve_identity and is_clean_xtts_reference(input_path, sr):
        return input_path
    source_path = input_path
    extracted_path = None
    cleaned_path = None
    if Path(input_path).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
        extracted_path = extract_audio_track(input_path, sr)
        source_path = extracted_path
    elif Path(input_path).suffix.lower() in {".mp3", ".m4a", ".ogg", ".ogx", ".opus", ".flac"}:
        cleaned_path = transcode_reference_with_ffmpeg(input_path, sr)
        source_path = cleaned_path
    try:
        audio, _ = _load_mono(source_path, sr)
        if audio.size and librosa is not None:
            audio = librosa.effects.trim(audio, top_db=35)[0]
        audio = _select_loud_reference_window(audio, sr, max_seconds)
        if audio.size:
            if not preserve_identity:
                try:
                    from scipy import signal

                    nyquist = sr / 2
                    low = 80 / nyquist
                    high = min(7600, nyquist - 100) / nyquist
                    if 0 < low < high < 1:
                        sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
                        audio = signal.sosfiltfilt(sos, audio).astype(np.float32)
                except Exception:
                    pass
            audio = audio - float(np.mean(audio))
            audio = normalize_audio(audio, peak=0.82)
        output_path = tempfile.mktemp(suffix="_ref.wav", dir=settings.temp_audio_dir)
        write_audio(output_path, audio, sr)
        return output_path
    finally:
        if extracted_path and os.path.exists(extracted_path):
            os.remove(extracted_path)
        if cleaned_path and cleaned_path != input_path and os.path.exists(cleaned_path):
            os.remove(cleaned_path)


def build_combined_reference_audio(
    input_paths: Iterable[str],
    output_path: str,
    sample_rate: Optional[int] = None,
    max_seconds_per_file: float = 22.0,
) -> str:
    """Create one clean reference WAV from multiple clips of the same speaker."""
    settings = get_settings()
    sr = sample_rate or settings.audio_sample_rate
    paths = [Path(path) for path in input_paths if path and Path(path).exists()]
    if not paths:
        raise FileNotFoundError("No reference audio files were found")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output_mtime = output.stat().st_mtime
        if all(path.stat().st_mtime <= output_mtime for path in paths):
            return str(output)

    cleaned_paths: list[str] = []
    chunks: list[np.ndarray] = []
    pause = np.zeros(int(sr * 0.18), dtype=np.float32)
    try:
        for path in paths:
            cleaned = clean_reference_with_ffmpeg(str(path), sr)
            if cleaned != str(path):
                cleaned_paths.append(cleaned)
            audio, _ = _load_mono(cleaned, sr)
            if audio.size and librosa is not None:
                audio = librosa.effects.trim(audio, top_db=32)[0]
            max_samples = int(sr * max_seconds_per_file)
            if audio.size > max_samples:
                audio = audio[:max_samples]
            if audio.size:
                audio = audio - float(np.mean(audio))
                chunks.append(normalize_audio(audio, peak=0.82))
                chunks.append(pause)
        if not chunks:
            raise RuntimeError("Reference audio files did not contain usable speech")
        merged = np.concatenate(chunks[:-1] if chunks[-1] is pause else chunks)
        write_audio(str(output), normalize_audio(merged, peak=0.9), sr)
        return str(output)
    finally:
        for path in cleaned_paths:
            if os.path.exists(path):
                os.remove(path)


def cleanup_temp_files(max_age_seconds: int = 3600) -> int:
    temp_dir = get_settings().temp_audio_dir
    if not temp_dir.exists():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for path in temp_dir.glob("*"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def write_silence(path: str, duration_seconds: float, sample_rate: Optional[int] = None) -> str:
    sr = sample_rate or get_settings().audio_sample_rate
    samples = max(1, int(duration_seconds * sr))
    write_audio(path, np.zeros(samples, dtype=np.float32), sr)
    return path


def _clean_clip_edges(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    if librosa is not None:
        audio = librosa.effects.trim(audio, top_db=30)[0].astype(np.float32)
    if audio.size == 0:
        return audio
    fade_samples = min(int(sample_rate * 0.035), audio.size // 4)
    if fade_samples > 1:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    return audio


def _compress_long_silences(
    audio: np.ndarray,
    sample_rate: int,
    max_silence_seconds: float = 0.12,
    top_db: float = 34.0,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    frame = max(1, int(sample_rate * 0.02))
    hop = max(1, int(sample_rate * 0.01))
    rms = []
    for start in range(0, max(1, audio.size - frame + 1), hop):
        chunk = audio[start:start + frame]
        rms.append(float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0)
    if not rms:
        return audio
    peak = max(rms)
    if peak <= 1e-6:
        return audio
    threshold = peak * (10 ** (-top_db / 20))
    voiced = np.array(rms) > threshold
    max_silence_frames = max(1, int(max_silence_seconds / 0.01))
    keep = np.zeros(audio.size, dtype=bool)
    silence_run = 0
    for index, is_voiced in enumerate(voiced):
        start = index * hop
        end = min(audio.size, start + frame)
        if is_voiced:
            silence_run = 0
            keep[start:end] = True
            continue
        silence_run += 1
        if silence_run <= max_silence_frames:
            keep[start:end] = True
    compressed = audio[keep]
    return compressed if compressed.size >= int(sample_rate * 0.25) else audio


def expected_dialogue_seconds(text: str, language: str = "en") -> float:
    language = (language or "en").strip().lower()
    if language in {"hi", "hindi", "hin"}:
        words = re.findall(r"[^\s,.;!?।]+", text or "")
        word_count = max(1, len(words))
        return min(18.0, max(3.0, word_count * 0.8 + 2.0))
    words = re.findall(r"\b[\w']+\b", text or "")
    word_count = max(1, len(words))
    return min(12.0, max(1.5, word_count * 0.45 + 1.0))


def stabilize_hindi_voice(input_path: str, sample_rate: Optional[int] = None) -> str:
    """Make Hindi XTTS output less trembly/weepy without changing words."""
    # Disabled bandpass filtering as it drastically changes vocal texture and speaker identity
    return input_path


def limit_dialogue_clip(input_path: str, text: str, sample_rate: Optional[int] = None, language: str = "en") -> str:
    sr = sample_rate or get_settings().audio_sample_rate
    normalized_language = (language or "en").strip().lower()
    audio, _ = _load_mono(input_path, sr)
    audio = _clean_clip_edges(audio, sr)
    audio = _compress_long_silences(audio, sr)
    max_samples = int(expected_dialogue_seconds(text, language=language) * sr)
    # Only cut clear over-generation. XTTS timing varies a lot across languages
    # and speakers, so a tight cap can chop valid words.
    trim_multiplier = 2.4 if normalized_language in {"hi", "hindi", "hin"} else 1.08
    trim_threshold = int(max_samples * trim_multiplier)
    if audio.size > trim_threshold:
        logger.info(
            "Trimming generated dialogue clip: language={}, original_seconds={:.2f}, capped_seconds={:.2f}, text={}",
            language,
            audio.size / sr,
            trim_threshold / sr,
            (text or "")[:80],
        )
        audio = audio[:trim_threshold]
        fade_samples = min(int(sr * 0.08), audio.size // 4)
        if fade_samples > 1:
            audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    write_audio(input_path, normalize_audio(audio, peak=0.9), sr)
    return input_path


def concatenate_wavs(paths: Iterable[str], output_path: str, pause_seconds: float = 0.35) -> str:
    paths = [path for path in paths if path and Path(path).exists()]
    if not paths:
        write_silence(output_path, 0.25)
        return output_path

    chunks = []
    sample_rate = get_settings().audio_sample_rate
    pause = np.zeros(int(sample_rate * pause_seconds), dtype=np.float32)
    for path in paths:
        audio, sr = _load_mono(path, sample_rate)
        audio = _clean_clip_edges(audio, sample_rate)
        chunks.append(normalize_audio(audio, peak=0.9))
        chunks.append(pause)
    merged = np.concatenate(chunks[:-1]) if chunks else pause
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_audio(output_path, normalize_audio(merged), sample_rate)
    return output_path


def export_audio(input_wav: str, output_path: str) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".wav":
        shutil.copyfile(input_wav, output)
        return str(output)
    if output.suffix.lower() == ".mp3":
        try:
            from pydub import AudioSegment
        except Exception as exc:
            raise RuntimeError("MP3 export requires pydub and FFmpeg") from exc
        AudioSegment.from_wav(input_wav).export(output, format="mp3", bitrate="192k")
        return str(output)
    raise ValueError("Only WAV and MP3 exports are supported")


def ffmpeg_available() -> bool:
    return shutil.which(get_settings().ffmpeg_path) is not None


def run_ffmpeg_normalize(input_path: str, output_path: str) -> str:
    settings = get_settings()
    if not ffmpeg_available():
        return preprocess_audio(input_path)
    subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            str(settings.audio_sample_rate),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def tone_preview(text: str, output_path: str, seconds: float = 1.2) -> str:
    sr = get_settings().audio_sample_rate
    freq = 180 + (sum(ord(ch) for ch in text) % 360)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    envelope = np.minimum(1.0, np.linspace(0, 12, t.size)) * np.minimum(1.0, np.linspace(12, 0, t.size))
    audio = 0.15 * np.sin(2 * math.pi * freq * t) * envelope
    write_audio(output_path, audio.astype(np.float32), sr)
    return output_path
