# Entry point for FastAPI backend


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from backend.core.config import get_settings
from backend.api import router as api_router
from backend.api.project import router as project_router
from backend.api.voice_profile import router as profile_router
import uvicorn
from loguru import logger


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0")

# Include API routers
app.include_router(api_router, prefix="/api")
app.include_router(project_router, prefix="/api/project")
app.include_router(profile_router, prefix="/api/profile")


@app.on_event("startup")
def startup_event():
    from backend.core.startup import run_startup_checks
    try:
        app.state.startup_status = run_startup_checks()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "offline": True,
        "startup": getattr(app.state, "startup_status", {}),
    }


@app.get("/ready")
def readiness_check():
    startup = getattr(app.state, "startup_status", {})
    return {
        "ready": True,
        "ffmpeg": startup.get("ffmpeg", {}).get("available", False),
        "xtts_backend": startup.get("xtts", {}).get("backend", "unknown"),
        "xtts_models": startup.get("models", {}).get("ready_xtts", []),
        "ui": f"http://localhost:{settings.gradio_port}",
    }

# Global error handler
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        raise


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on {} {}", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host=settings.host, port=settings.port, reload=True)
