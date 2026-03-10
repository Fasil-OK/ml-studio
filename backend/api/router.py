from fastapi import APIRouter

from api.projects import router as projects_router
from api.datasets import router as datasets_router
from api.experiments import router as experiments_router
from api.models import router as models_router
from api.training import router as training_router
from api.evaluation import router as evaluation_router
from api.explanations import router as explanations_router
from api.inference import router as inference_router
from api.resources import router as resources_router
from api.chat import router as chat_router
from api.hpo import router as hpo_router
from api.settings import router as settings_router

api_router = APIRouter()

api_router.include_router(projects_router, prefix="/projects", tags=["projects"])
api_router.include_router(datasets_router, tags=["datasets"])
api_router.include_router(experiments_router, tags=["experiments"])
api_router.include_router(models_router, prefix="/models", tags=["models"])
api_router.include_router(training_router, tags=["training"])
api_router.include_router(evaluation_router, tags=["evaluation"])
api_router.include_router(explanations_router, tags=["explanations"])
api_router.include_router(inference_router, tags=["inference"])
api_router.include_router(resources_router, prefix="/resources", tags=["resources"])
api_router.include_router(chat_router, tags=["chat"])
api_router.include_router(hpo_router, tags=["hpo"])
api_router.include_router(settings_router, prefix="/settings", tags=["settings"])
