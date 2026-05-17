from pathlib import Path
import tempfile
import os

from fastapi import APIRouter, Body, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from backend.core.config import get_settings
from backend.pipeline.xtts_pipeline import (
    generate_conversation_from_script,
    generate_two_voice_conversation,
    generate_voice,
    get_available_styles,
    generate_multi_speaker,
)

router = APIRouter()


async def _save_upload(upload: UploadFile) -> str:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Audio upload is required")
    suffix = Path(upload.filename).suffix or ".wav"
    fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=get_settings().temp_audio_dir)
    os.close(fd)
    with open(temp_path, "wb") as handle:
        handle.write(await upload.read())
    return temp_path


@router.post("/conversation")
async def conversation(
    script: str = Form(...),
    voice_1: UploadFile = File(...),
    voice_2: UploadFile = File(...),
    voice_3: UploadFile | None = File(None),
    speaker_1_style: str = Form("original"),
    speaker_2_style: str = Form("original"),
    speaker_3_style: str = Form("original"),
    speaker_1_speed: float = Form(1.0),
    speaker_2_speed: float = Form(1.0),
    speaker_3_speed: float = Form(1.0),
    speaker_2_backend: str = Form("local"),
    language: str = Form("en"),
    pause_seconds: float = Form(0.05),
):
    temp_paths: list[str] = []
    try:
        temp_paths.append(await _save_upload(voice_1))
        temp_paths.append(await _save_upload(voice_2))
        voice_3_path = None
        if voice_3 is not None and voice_3.filename:
            voice_3_path = await _save_upload(voice_3)
            temp_paths.append(voice_3_path)
        output_path = generate_conversation_from_script(
            script=script,
            voice_1_path=temp_paths[0],
            voice_2_path=temp_paths[1],
            voice_3_path=voice_3_path,
            speaker_1_style=speaker_1_style,
            speaker_2_style=speaker_2_style,
            speaker_3_style=speaker_3_style,
            speaker_1_speed=speaker_1_speed,
            speaker_2_speed=speaker_2_speed,
            speaker_3_speed=speaker_3_speed,
            speaker_2_backend=speaker_2_backend,
            language=language,
            pause_seconds=pause_seconds,
        )
        return JSONResponse({"audio_path": output_path})
    except Exception as e:
        logger.exception("Conversation endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

@router.post("/generate")
async def generate(
    text: str = Form(...),
    style: str = Form("original"),
    speaker_wav: UploadFile = File(...),
    output_filename: str = Form(None),
    language: str = Form("en"),
    speed: float = Form(1.0),
):
    try:
        if not speaker_wav.filename:
            raise HTTPException(status_code=400, detail="speaker_wav is required")
        temp_path = await _save_upload(speaker_wav)
        result = generate_voice(text, temp_path, style, output_filename, language=language, speed=speed)
        os.remove(temp_path)
        return JSONResponse(result)
    except Exception as e:
        logger.exception("XTTS clone endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clone")
async def clone(
    text: str = Form(...),
    speaker_wav: UploadFile = File(...),
    output_filename: str = Form(None),
    language: str = Form("en"),
    speed: float = Form(1.0),
):
    return await generate(
        text=text,
        style="original",
        speaker_wav=speaker_wav,
        output_filename=output_filename,
        language=language,
        speed=speed,
    )

@router.post("/multi-speaker")
async def multi_speaker(dialogues: list = Body(...), pause_seconds: float = 0.35):
    try:
        if not dialogues:
            raise HTTPException(status_code=400, detail="At least one dialogue row is required")
        output_path = generate_multi_speaker(dialogues, pause_seconds=pause_seconds)
        return JSONResponse({"audio_path": output_path})
    except Exception as e:
        logger.exception("XTTS multi-speaker endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/styles")
async def styles():
    return JSONResponse({"styles": get_available_styles()})
