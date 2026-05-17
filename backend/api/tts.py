from fastapi import APIRouter, Form
from backend.services.tts import generate_tts

router = APIRouter()

@router.post("/generate")
async def tts_generate(
    text: str = Form(...),
    speaker_name: str = Form(...)
):
    output_path = await generate_tts(text, speaker_name)
    return {"audio_path": output_path}
