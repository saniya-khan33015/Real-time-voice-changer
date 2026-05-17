# Multi-Voice AI Voice Studio

## Features
- Create projects with multiple AI speakers
- Assign cloned voices to dialogue lines
- Compose scripts for conversations, podcasts, dubbing, etc.
- Save/load projects locally
- Manage voice profiles (save, rename, delete, load)
- Generate and export full multi-speaker audio timelines
- 100% offline/local, no cloud APIs

## How to Use
1. Start backend: `python backend/app.py`
2. Start multi-voice UI: `python frontend/gradio_ui/multivoice_ui.py`
3. In the UI:
   - Create voice profiles (via main UI)
   - Create a new project: enter script, assign speakers
   - Generate audio for the project
   - Download/export the result

## Script Example
```
A: Hello, how are you?
B: I'm great! Ready for the podcast?
A: Absolutely, let's start.
```

## Speaker Mapping Example
```
{"A": "profile1", "B": "profile2"}
```

## All processing is local/offline. Add your own TTS/voice models for best results.
