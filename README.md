# XTTS Voice Cloning Web App

Python-only, offline-first web app for direct local TTS voice cloning.

The app uses direct local TTS voice cloning. It does not use voice conversion, `.index` models, or a post-generation voice changer.

Speaker 1 and Speaker 2 both use local voice cloning from uploaded reference audio. The default backend is local XTTS/Coqui so the app runs immediately. A self-hosted Fish Speech server can be enabled later for better speaker similarity without a third-party API key.

## What Is Implemented

- FastAPI backend with an XTTS voice cloning endpoint
- Gradio UI with two speaker uploads, script input, generate button, audio player, and download option
- Local Fish Speech backend through a self-hosted `POST /v1/tts` server
- Dynamic local XTTS model discovery under `ai_models/xtts` and `ai_models/tts`
- Cache-aware model loading and unload hooks
- Audio preprocessing, normalization, mono conversion, FFmpeg validation, and fallback WAV I/O
- Sentence-sized XTTS turn generation to reduce long-line voice drift
- Offline dummy TTS fallback so the platform boots before real XTTS models are installed

The dummy fallback is for development only. Natural cloned speech requires a local XTTS/Coqui model.

## Folder Layout

```text
ai_models/
  xtts/
    xtts_v2/
      config.json
      model.pth
      speakers_xtts.pth
      vocab.json
  tts/
    your_local_coqui_model/
      config.json
      model.pth
  cloned_voices/
    user_voice_1/
      samples/reference.wav
      embedding.json
      metadata.json
      preview.wav
audio/
  references/
    speaker1_rahul_clean.wav
  downloads/
    generated_conversation_*.wav
  temp/
  outputs/
projects/
  project_name/
    script.txt
    voices.json
    generated_audio/
    final_mix.wav
    project.json
```

## Execution Guide

### 1. Environment Setup

Use Python 3.10 for the AI stack.

```powershell
cd "F:\REAL TIME VOICE CHANGER"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Install FFmpeg and make sure `ffmpeg` is on PATH, or set `FFMPEG_PATH` in `.env`.

Check FFmpeg:

```powershell
ffmpeg -version
```

### 2. Model Setup

Place the XTTS model locally. No cloud or paid APIs are required.

```text
ai_models/
  xtts/
    local_xtts_model/
      config.json
      model.pth
  cloned_voices/
    demo_voice/
      samples/reference.wav
      metadata.json
      preview.wav
```

The app discovers model folders dynamically at startup. If XTTS is missing, the app still boots with a local dummy TTS fallback so the UI and API can be demonstrated.

### 3. Backend Execution

Backend:

```powershell
cd "F:\REAL TIME VOICE CHANGER"
.\venv\Scripts\activate
python -m backend.app
```

Expected startup logs include FFmpeg, Torch/CUDA status, XTTS backend, API routes, and frontend URL.

### Fish Speech Backend

For better speaker similarity than XTTS, run a self-hosted Fish Speech server locally, then set `TTS_BACKEND=fish_speech` in `.env`.

```env
TTS_BACKEND=fish_speech
FISH_SPEECH_URL=http://127.0.0.1:8080/v1/tts
FISH_SPEECH_TEMPERATURE=0.25
FISH_SPEECH_TOP_P=0.55
FISH_SPEECH_REPETITION_PENALTY=1.25
FISH_SPEECH_SEED=42
FISH_SPEECH_FALLBACK_TO_XTTS=true

```

Official Fish Speech server command shape:

```powershell
python tools/api_server.py `
  --llama-checkpoint-path checkpoints/s2-pro `
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth `
  --listen 0.0.0.0:8080
```

The app calls `http://127.0.0.1:8080/v1/tts` with the uploaded speaker reference audio. This is local/self-hosted and does not use the hosted Fish Audio API key.

Check the local Fish Speech server before generation:

```powershell
cd "F:\REAL TIME VOICE CHANGER"
.\venv\Scripts\activate
python scripts\check_fish_speech_server.py
```

If this check fails, start the Fish Speech server first. If `FISH_SPEECH_FALLBACK_TO_XTTS=true`, generation falls back to XTTS when the local Fish Speech server is unavailable.

### 4. Frontend Execution

Open a second PowerShell terminal:

Gradio UI:

```powershell
cd "F:\REAL TIME VOICE CHANGER"
.\venv\Scripts\activate
python -m frontend.gradio_ui.app
```

Open:

- API docs: `http://localhost:8001/docs`
- Health: `http://localhost:8001/health`
- Readiness: `http://localhost:8001/ready`
- UI: `http://localhost:7860`

### 5. Example Workflows

1. Upload clean 15-60 second reference voices for Speaker 1 and Speaker 2.
2. Enter a dialogue script using `Speaker 1:` and `Speaker 2:` labels.
3. Click Generate Conversation.
4. Play or download the generated WAV.

The UI does not preload any default voices. Upload or record the exact Speaker 1 and Speaker 2 reference audio you want for each generation.

In the Gradio conversation flow, both speakers are generated locally from the uploaded reference audio only. The backend does not replace Speaker 1 or Speaker 2 with any hidden default file. The generated WAV is copied to `audio/downloads/` and shown in the download field.

`C:\Users\khans\Downloads\document_6068879421647889274.mp4` is only the quality reference for the desired Hindi smoothness and natural conversational flow. It is not used as Speaker 1 or Speaker 2, and it is not used for conversion.

### 6. Troubleshooting

- `ModuleNotFoundError: No module named 'backend'`: run commands from the project root and prefer `python -m backend.app` / `python -m frontend.gradio_ui.app`.
- `Permission denied` while creating `venv`: do not recreate `venv` while it is activated. Close terminals using it, then recreate only if needed.
- No natural speech: add a local XTTS/Coqui model under `ai_models/xtts/`. Without it, the dummy local TTS fallback is used.
- MP3 export fails: confirm `ffmpeg -version` works or set `FFMPEG_PATH` in `.env`.

## API Highlights

- `GET /api/model/list`
- `POST /api/tts/generate`
- `POST /api/xtts/clone`
- `POST /api/xtts/generate`
- `POST /api/xtts/conversation`
- `POST /api/project/create`
- `GET /api/project/list`
- `GET /api/project/load`
- `POST /api/project/generate_audio`
- `POST /api/project/export`
- `GET /api/profile/list`
- `POST /api/profile/rename`
- `DELETE /api/profile/delete`

## Reference Audio Tips

- Use WAV or MP3 when possible.
- Use 15-60 seconds of clean speech.
- Use one speaker only.
- Avoid music, echo, heavy compression, and background noise.
- For Hindi cloning, keep the speaker speed at `1.0` first. Changing speed can make XTTS drift away from the reference voice more than it does in English.

## Two Speaker Script Format

```text
Speaker 1: Hello, how are you?
Speaker 2: I am fine. Today we are testing XTTS voice cloning.
Speaker 1: Great, this keeps both voices separate.
```

## Testing

```powershell
python -m pytest tests
```

## Production Notes

- Keep all models local in `ai_models/`
- Enable CUDA PyTorch for GPU acceleration
- Tune `MODEL_CACHE_SIZE` and `USE_FP16` in `.env`
- Use a local XTTS/Coqui model folder for natural cloned speech generation
