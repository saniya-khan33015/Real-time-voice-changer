from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urljoin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import requests

from backend.core.config import get_settings


settings = get_settings()
API_BASE_URL = f"http://localhost:{settings.port}"
API_URL = urljoin(API_BASE_URL, "/api/xtts/conversation")
SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".ogx", ".opus", ".mp4", ".mov", ".mkv", ".webm"}
DOWNLOAD_OUTPUT_DIR = PROJECT_ROOT / "audio" / "downloads"
LANGUAGE_OPTIONS = {"English": "en", "Hindi": "hi"}


def _existing_audio_path(*candidates: str | Path | None) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate).strip().strip("\"'"))
        if path.exists():
            return str(path)
    return None


def prepare_download_copy(audio_path: str | None) -> str | None:
    if not audio_path:
        return None
    source = Path(audio_path)
    if not source.exists():
        return None
    DOWNLOAD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = DOWNLOAD_OUTPUT_DIR / f"generated_conversation_{int(time.time())}.wav"
    shutil.copyfile(source, target)
    return str(target)


def _format_error(response: requests.Response) -> str:
    if response.status_code == 404:
        return (
            f"Error: backend endpoint not found at {API_URL}. "
            "Restart the FastAPI backend and confirm it is running from this project."
        )
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text
    return f"Error: {detail or response.text}"


def _validate_two_speaker_script(script: str) -> str | None:
    for raw in (script or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            return f"Line must start with Speaker 1: or Speaker 2:: {line}"
        speaker, text = line.split(":", 1)
        speaker = speaker.strip().lower().replace(" ", "")
        if speaker not in {"speaker1", "speaker2"}:
            return f"Only Speaker 1 and Speaker 2 labels are allowed: {line}"
        if not text.strip():
            return f"Text is missing after speaker label: {line}"
    return None


def generate_two_speaker_conversation(
    voice_1,
    voice_2,
    language,
    script,
    pause_seconds,
    speaker_1_speed,
    speaker_2_speed,
):
    if not voice_1 or not voice_2 or not script:
        return None, None, None, "Upload both speaker voices and enter a script."
    script_error = _validate_two_speaker_script(script)
    if script_error:
        return None, None, None, script_error
    for label, path in (("Speaker 1 Voice", voice_1), ("Speaker 2 Voice", voice_2)):
        if Path(path).suffix.lower() not in SUPPORTED_AUDIO:
            return None, None, None, f"{label} must be WAV, MP3, FLAC, M4A, OGG, OGX, OPUS, MP4, MOV, MKV, or WEBM."
    try:
        with open(voice_1, "rb") as first, open(voice_2, "rb") as second:
            response = requests.post(
                API_URL,
                data={
                    "script": script,
                    "language": LANGUAGE_OPTIONS.get(language, "en"),
                    "pause_seconds": pause_seconds,
                    "speaker_1_speed": speaker_1_speed,
                    "speaker_2_speed": speaker_2_speed,
                    "speaker_2_backend": "local",
                },
                files={
                    "voice_1": (Path(voice_1).name, first),
                    "voice_2": (Path(voice_2).name, second),
                },
                timeout=900,
            )
        if response.status_code != 200:
            return None, None, None, _format_error(response)
        result = response.json()
        audio_path = result.get("audio_path")
        download_path = prepare_download_copy(audio_path)
        if not download_path:
            return audio_path, None, None, f"Done, but download copy could not be prepared. Output path: {audio_path}"
        return audio_path, download_path, download_path, "Done. Download file is ready."
    except Exception as exc:
        return None, None, None, f"Error: {exc}"


def build_ui():
    default_voice_1 = _existing_audio_path(
        settings.default_speaker_1_reference_audio,
        PROJECT_ROOT / "Rahul_Gandhi_Cleaned_Final.wav",
    )
    default_voice_2 = _existing_audio_path(
        settings.default_speaker_2_reference_audio,
        PROJECT_ROOT / "Narendra_Modi_voice.ogg",
    )

    with gr.Blocks(title="Two Person TTS Voice Cloning Conversation") as demo:
        gr.Markdown("# Two Person TTS Voice Cloning Conversation")
        with gr.Row():
            voice_1 = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Speaker 1 Reference Voice",
                value=default_voice_1,
            )
            voice_2 = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Speaker 2 Reference Voice",
                value=default_voice_2,
            )
        language = gr.Radio(["English", "Hindi"], value="Hindi", label="Language")
        script = gr.Textbox(
            label="Dialogue / Script",
            lines=10,
            value="Speaker 1: नमस्ते, आज हम एआई वॉइस क्लोनिंग के बारे में बात करेंगे।\nSpeaker 2: बिल्कुल, यह दो अलग आवाजों में नैचुरल बातचीत जनरेट करेगा।\nSpeaker 1: इसमें सिर्फ टीटीएस वॉइस क्लोनिंग इस्तेमाल होगी, वॉइस कन्वर्जन नहीं।",
        )
        with gr.Row():
            speaker_1_speed = gr.Slider(0.9, 1.35, value=1.0, step=0.02, label="Speaker 1 Speed")
            speaker_2_speed = gr.Slider(0.9, 1.35, value=1.0, step=0.02, label="Speaker 2 Speed")
            pause_seconds = gr.Slider(0.0, 0.4, value=0.05, step=0.01, label="Pause Between Lines")
        generate_btn = gr.Button("Generate Conversation", variant="primary")
        output_audio = gr.Audio(label="Generated Two Speaker Audio")
        download_button = gr.DownloadButton(label="Download WAV", value=None)
        download_audio = gr.File(label="Download Generated WAV")
        status = gr.Textbox(label="Status")

        generate_btn.click(
            generate_two_speaker_conversation,
            inputs=[voice_1, voice_2, language, script, pause_seconds, speaker_1_speed, speaker_2_speed],
            outputs=[output_audio, download_button, download_audio, status],
            show_progress=True,
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=get_settings().gradio_port)
