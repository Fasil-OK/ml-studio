from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

ENV_PATH = Path(__file__).parent.parent / ".env"


class LLMSettings(BaseModel):
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"


@router.get("")
async def get_settings():
    settings = _read_env()
    # Mask API key
    key = settings.get("LLM_API_KEY", "")
    masked = key[:4] + "***" + key[-4:] if len(key) > 8 else "***" if key else ""
    return {
        "llm_base_url": settings.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        "llm_api_key_masked": masked,
        "llm_model": settings.get("LLM_MODEL", "gpt-4o"),
        "has_key": bool(key),
    }


@router.put("")
async def update_settings(data: LLMSettings):
    env_vars = _read_env()
    env_vars["LLM_BASE_URL"] = data.llm_base_url
    if data.llm_api_key:  # Only update if provided (not empty)
        env_vars["LLM_API_KEY"] = data.llm_api_key
    env_vars["LLM_MODEL"] = data.llm_model
    _write_env(env_vars)

    # Reload config
    from config import settings as app_settings
    app_settings.llm_base_url = data.llm_base_url
    if data.llm_api_key:
        app_settings.llm_api_key = data.llm_api_key
    app_settings.llm_model = data.llm_model

    return {"status": "updated"}


def _read_env() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def _write_env(env_vars: dict):
    lines = [f"{k}={v}" for k, v in env_vars.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n")
