from fastapi import APIRouter, UploadFile, File, Form
from backend.services.project_manager import (
    create_project, load_project, save_project, delete_project, list_projects, generate_project_audio, export_project
)

router = APIRouter()

@router.post("/create")
async def api_create_project(name: str = Form(...), script: str = Form(...), speakers: str = Form(...)):
    return create_project(name, script, speakers)

@router.get("/list")
async def api_list_projects():
    return {"projects": list_projects()}

@router.get("/load")
async def api_load_project(name: str):
    return load_project(name)

@router.post("/save")
async def api_save_project(name: str = Form(...), data: str = Form(...)):
    return save_project(name, data)

@router.delete("/delete")
async def api_delete_project(name: str):
    return delete_project(name)

@router.post("/generate_audio")
async def api_generate_project_audio(name: str = Form(...), pause_seconds: float = Form(0.35)):
    return generate_project_audio(name, pause_seconds=pause_seconds)

@router.post("/export")
async def api_export_project(name: str = Form(...), fmt: str = Form("wav")):
    return export_project(name, fmt)
