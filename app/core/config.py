import sys
from typing import Annotated

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "fastapi-vector-db"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    model_config = {"env_prefix": "APP_"}


settings = Settings()

# Configure Loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)
