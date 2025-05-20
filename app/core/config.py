import sys

from loguru import logger
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "fastapi-vector-db"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    
    # Persistence settings
    ENABLE_PERSISTENCE: bool = False
    PERSISTENCE_PATH: str = "./data/vector_db.json"
    PERSISTENCE_INTERVAL: int = 300  # Save every 5 minutes

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
