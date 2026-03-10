from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from database import init_db
from api.router import api_router
from ws.training_ws import router as training_ws_router
from ws.chat_ws import router as chat_ws_router
from ws.hpo_ws import router as hpo_ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    for d in [
        settings.upload_dir,
        settings.dataset_dir,
        settings.checkpoint_dir,
        settings.export_dir,
        settings.explanation_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    await init_db()
    yield
    # Shutdown


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API
app.include_router(api_router, prefix="/api")

# WebSocket endpoints
app.include_router(training_ws_router)
app.include_router(chat_ws_router)
app.include_router(hpo_ws_router)

# Serve stored files (explanations, samples, etc.)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")
