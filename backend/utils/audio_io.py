from pathlib import Path

import numpy as np

try:
    import soundfile as _sf
except Exception:
    _sf = None


def read_audio(path: str, dtype: str = "float32", always_2d: bool = False):
    if _sf is not None:
        return _sf.read(path, dtype=dtype, always_2d=always_2d)
    from scipy.io import wavfile

    sr, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if audio.dtype.kind in {"i", "u"}:
        audio = audio.astype(np.float32) / max(float(np.iinfo(audio.dtype).max), 1.0)
    else:
        audio = audio.astype(np.float32)
    if always_2d and audio.ndim == 1:
        audio = audio[:, None]
    return audio.astype(dtype), sr


def write_audio(path: str, audio, sample_rate: int) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if _sf is not None:
        _sf.write(path, audio, sample_rate)
        return path
    from scipy.io import wavfile

    wavfile.write(path, sample_rate, np.clip(audio, -1.0, 1.0))
    return path
