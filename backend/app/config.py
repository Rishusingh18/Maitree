import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field, field_validator

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

        APP_NAME: str = Field(default="MetroSafetyAPI", description="Application name")
        DEBUG: bool = Field(default=False, description="Enable debug mode")
        LOG_LEVEL: str = Field(default="info", description="Logging level")
        HOST: str = Field(default="0.0.0.0", description="Host address")
        PORT: int = Field(default=8000, description="Port to listen on")
        API_KEY: str = Field(default="", description="Simple API key for auth")
        CROSSING_MODEL_PATH: str = Field(default="backend/models/crossing_model.pt", description="Crossing model path")
        GESTURE_MODEL_PATH: str = Field(default="backend/models/gesture_model.pt", description="Gesture model path")
        BEHAVIOUR_MODEL_PATH: str = Field(default="backend/models/behaviour_model.pt", description="Behaviour model path")
        MAX_WORKERS: int = Field(default=4, description="Thread pool size")

        @field_validator("LOG_LEVEL")
        @classmethod
        def _validate_log_level(cls, v: str) -> str:
            allowed = {"debug", "info", "warning", "error", "critical"}
            return v.lower() if v.lower() in allowed else "info"

    settings = Settings()

except Exception:
    # Graceful fallback to standard dataclass / os.getenv
    class Settings:
        APP_NAME: str = os.getenv("APP_NAME", "MetroSafetyAPI")
        DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")
        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", "8000"))
        API_KEY: str = os.getenv("API_KEY", "")
        CROSSING_MODEL_PATH: str = os.getenv("CROSSING_MODEL_PATH", "backend/models/crossing_model.pt")
        GESTURE_MODEL_PATH: str = os.getenv("GESTURE_MODEL_PATH", "backend/models/gesture_model.pt")
        BEHAVIOUR_MODEL_PATH: str = os.getenv("BEHAVIOUR_MODEL_PATH", "backend/models/behaviour_model.pt")
        MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))

    settings = Settings()
