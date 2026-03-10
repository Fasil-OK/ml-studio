from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "AutoML Platform"
    debug: bool = True

    # Database
    database_url: str = "sqlite+aiosqlite:///./storage/automl.db"

    # Storage paths
    storage_dir: Path = Path("storage")
    upload_dir: Path = Path("storage/uploads")
    dataset_dir: Path = Path("storage/datasets")
    checkpoint_dir: Path = Path("storage/checkpoints")
    export_dir: Path = Path("storage/exports")
    explanation_dir: Path = Path("storage/explanations")

    # LLM Configuration (OpenAI-compatible)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"

    # Training
    max_concurrent_training: int = 1
    default_device: str = "cuda"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
