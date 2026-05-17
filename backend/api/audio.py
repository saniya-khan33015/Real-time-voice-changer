from fastapi import APIRouter, UploadFile, File
from backend.services.audio_processing import process_audio
from fastapi.responses import FileResponse
import os

router = APIRouter()

@router.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    processed_path = await process_audio(audio)
    return FileResponse(processed_path, filename=os.path.basename(processed_path))
