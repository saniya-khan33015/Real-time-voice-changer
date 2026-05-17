from fastapi import APIRouter
from . import tts, model, audio
from . import routes as xtts_routes

router = APIRouter()
router.include_router(tts.router, prefix="/tts", tags=["Text-to-Speech"])
router.include_router(model.router, prefix="/model", tags=["Model Management"])
router.include_router(audio.router, prefix="/audio", tags=["Audio Processing"])
router.include_router(xtts_routes.router, prefix="/xtts", tags=["XTTS Voice Cloning"])
