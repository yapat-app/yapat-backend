"""
Configuration
"""

from typing import Optional, Union

from pydantic_settings import BaseSettings


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
    DATA_ROOT: Optional[str] = "/data"
    HOST_DATA_ROOT: str | None = None  # host path (optional)
    HOST_MODELS_AL: str | None = None  # host path for models; used by docker-compose for mounts only
    
    # CORS - can be set as comma-separated string or list
    BACKEND_CORS_ORIGINS: Union[str, list] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # Mock data settings (for development/testing)
    USE_MOCK_AUDIO: bool = True
    MOCK_AUDIO_PATH: str = "/mock/audio"
    REAL_AUDIO_PATH: str = "/data/audio"

    @property
    def cors_origins_list(self) -> list:
        """Parse CORS origins from string or return list"""
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]
        return self.BACKEND_CORS_ORIGINS

    @property
    def audio_base_path(self) -> str:
        """Get audio path based on mock setting"""
        return self.MOCK_AUDIO_PATH if self.USE_MOCK_AUDIO else self.REAL_AUDIO_PATH

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 3600  # 1 hour max per task
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3300  # 55 minutes limit

    # OE_YAPAT Service (Custom Taxonomy Generation)
    OE_YAPAT_SERVICE_URL: str = "http://localhost:8002"  
    OE_YAPAT_API_KEY: Optional[str] = None
    OE_YAPAT_TIMEOUT: int = 60  # seconds
    OE_YAPAT_RETRY_ATTEMPTS: int = 3
    
    # WSSED GPU Server
    WSSED_GPU_SERVER_URL: str = "http://localhost:8003"  # URL of GPU server running WSSED
    WSSED_TIMEOUT: int = 300  # seconds (5 minutes for long operations)
    WSSED_POLL_INTERVAL: int = 10  # seconds between status polls
    
    # Active Learning - Species Models
    ACTIVE_LEARNING_MODELS_DIR: Optional[str] = None  # Directory containing pre-trained species models
    AUTO_REGISTER_SPECIES_MODELS: bool = True  # Automatically register species models for WEAKLY_LABELED datasets

    # PAM Active Learning
    PAM_AUTO_RETRAIN_THRESHOLD: int = 5  # Auto-retrain after N feedback events
    PAM_DEFAULT_DEVICE: str = "cpu"      # Default device for PAM inference/training
    PAM_BASE_MODEL_PATH: str = "models_AL/pam/base/base_pam_model.pt"  # Physical base model file
    PAM_CHECKPOINTS_DIR: str = "models_AL/pam/checkpoints"              # Versioned checkpoint storage

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
