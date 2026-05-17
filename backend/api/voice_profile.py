from fastapi import APIRouter, Form
from backend.services.voice_profile_manager import (
    list_profiles, save_profile, rename_profile, delete_profile, load_profile
)

router = APIRouter()

@router.get("/list")
async def api_list_profiles():
    return {"profiles": list_profiles()}

@router.post("/save")
async def api_save_profile(name: str = Form(...), data: str = Form(...)):
    return save_profile(name, data)

@router.post("/rename")
async def api_rename_profile(old_name: str = Form(...), new_name: str = Form(...)):
    return rename_profile(old_name, new_name)

@router.delete("/delete")
async def api_delete_profile(name: str):
    return delete_profile(name)

@router.get("/load")
async def api_load_profile(name: str):
    return load_profile(name)
