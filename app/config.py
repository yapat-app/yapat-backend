"""
Configuration
"""

from pydantic_settings import BaseSettings
from typing import Optional, Union
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    DATABASE_URL: str = "postgresql://yapat_user:yapat_password@localhost/yapat"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API
    API_STR: str = "/api"
    PROJECT_NAME: str = "YAPAT Backend"
    
    # CORS - can be set as comma-separated string or list
    BACKEND_CORS_ORIGINS: Union[str, list] = ["http://localhost:3000", "http://localhost:8000"]
    
    @property
    def cors_origins_list(self) -> list:
        """Parse CORS origins from string or return list"""
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]
        return self.BACKEND_CORS_ORIGINS
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 3600  # 1 hour max per task
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3300  # 55 minutes limit
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

