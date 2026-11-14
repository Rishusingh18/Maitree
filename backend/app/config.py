import os
from pydantic import BaseSettings, Field, validator

class Settings(BaseSettings):
    """Application configuration loaded from environment variables.
    All settings have sensible defaults for local development.
    """

    # General
    APP_NAME: str = Field(default="MetroSafetyAPI", description="Application name")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="info", description="Logging level")

    # Server
    HOST: str = Field(default="0.0.0.0", description="Host address")
    PORT: int = Field(default=8000, description="Port to listen on")

    # Security
    API_KEY: str = Field(default="", description="Simple API key for auth. Set in .env for production")

    # Model paths
    CROSSING_MODEL_PATH: str = Field(default="models/crossing_model.pt", description="Path to crossing detection model")
    GESTURE_MODEL_PATH: str = Field(default="models/gesture_model.pt", description="Path to gesture / threat detection model")

    # Inference settings
    MAX_WORKERS: int = Field(default=4, description="Thread pool size for async inference")

    @validator("LOG_LEVEL")
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid LOG_LEVEL: {v}. Choose from {allowed}")
        return v.lower()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate a singleton settings object that can be imported elsewhere
settings = Settings()
