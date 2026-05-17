# Environment Setup Instructions for XTTS Voice Cloning Studio

## 1. Python Environment
- Python 3.8+ recommended
- Create a virtual environment:
  ```sh
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

## 2. Install Requirements
- Install all dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## 3. FFmpeg
- FFmpeg must be installed and available in your PATH.
- Download from: https://ffmpeg.org/download.html
- Verify installation:
  ```sh
  ffmpeg -version
  ```

## 4. Torch Audio Backend
- Ensure your system supports torchaudio and torch for your hardware (CPU/GPU).
- For GPU (optional):
  ```sh
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

## 5. Run Backend
- Start FastAPI backend:
  ```sh
  uvicorn backend.app:app --reload
  ```

## 6. Model Placement
- Place XTTS models/checkpoints under `ai_models/xtts/` or `ai_models/tts/`.
- This project uses direct XTTS voice cloning only.

## 7. Test
- Use the provided API endpoints (see backend/api/routes.py) or test scripts.

## 8. Troubleshooting
- Ensure all dependencies are installed.
- Check logs for errors.
- Confirm model files exist and are accessible.

---
For further help, see README.md or contact the project maintainer.
