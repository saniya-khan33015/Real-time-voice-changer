from fastapi import APIRouter, Form
from backend.services.model_manager import list_models, switch_model

router = APIRouter()

@router.get("/list")
async def model_list():
    return {"models": list_models()}

@router.post("/switch")
async def model_switch(model_name: str = Form(...)):
    return switch_model(model_name)
