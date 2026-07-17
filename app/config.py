"""
Configuration
"""

from typing import Optional, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    LOG_LEVEL: str = "INFO"
    ENABLE_DOCS: bool = False  # Disable Swagger/ReDoc in production; set to true in dev
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

    # run_embedding dispatches one chord child per this many recordings,
    # instead of one per recording. Keeps chord/broker bookkeeping (Redis
    # counters, task messages) proportional to worker concurrency rather
    # than dataset size — a dataset with tens of thousands of recordings
    # would otherwise create tens of thousands of chord entries.
    EMBEDDING_CHORD_CHUNK_SIZE: int = 25

    # scan_dataset gets its own, longer time budget instead of the app-wide
    # default above — it can legitimately run for hours on very large
    # datasets (hundreds of GB, tens of thousands of files).
    SCAN_TASK_TIME_LIMIT: int = 90000       # 25 h hard kill
    SCAN_TASK_SOFT_TIME_LIMIT: int = 86400  # 24 h soft limit

    # OE_YAPAT Service (Custom Taxonomy Generation)
    OE_YAPAT_SERVICE_URL: str = "http://localhost:8002"  
    OE_YAPAT_API_KEY: Optional[str] = None
    OE_YAPAT_TIMEOUT: int = 120  # seconds — needs to cover up to 6 LLM calls in oe_yapat http_api_server
    OE_YAPAT_RETRY_ATTEMPTS: int = 2  # 2 retries max; 3× is too long when timeout is 120s
    
    # WSSED GPU Server
    WSSED_GPU_SERVER_URL: str = "http://localhost:8003"  # URL of GPU server running WSSED
    WSSED_TIMEOUT: int = 300  # seconds (5 minutes for long operations)
    WSSED_POLL_INTERVAL: int = 10  # seconds between status polls
    # Host path to WSSED focal-data outputs (for copying checkpoints into PAM_CHECKPOINTS_DIR)
    WSSED_FOCAL_DATA_ROOT: Optional[str] = None
    
    # Active Learning - Species Models
    ACTIVE_LEARNING_MODELS_DIR: Optional[str] = None  # Directory containing pre-trained species models
    AUTO_REGISTER_SPECIES_MODELS: bool = True  # Automatically register species models for FOCAL_RECORDINGS datasets

    # Feature Projection View (dataset-level dimensionality reduction)
    # Per-method point caps -- methods are gated independently rather than
    # rejecting the whole FPV run, since PCA stays cheap/fast at any n.
    # PCA has no cap.
    # UMAP runs on the main thread rather than a ThreadPoolExecutor thread
    # (see _compute_visualizations), so it has no fork-safety issue, but it
    # is single-threaded and therefore slow at large n: a multi-million-point
    # fit can exceed the Celery task time limit before finishing. Capped here
    # pending profiling against real task time limits.
    FPV_UMAP_MAX_POINTS: int = 100_000
    # t-SNE via openTSNE (FIt-SNE) is FFT-accelerated and memory-light
    # (~1.3GB at 20k), so it gets a high cap. Isomap builds a dense O(n^2)
    # geodesic-distance matrix and eigendecomposition; measured to OOM-kill a
    # 16GB worker at ~20k points, so it stays conservative.
    FPV_TSNE_MAX_POINTS: int = 50_000
    FPV_ISOMAP_MAX_POINTS: int = 15_000

    # PAM Active Learning
    PAM_AUTO_RETRAIN_THRESHOLD: int = 5  # Auto-retrain after N feedback events
    PAM_DEFAULT_DEVICE: str = "cpu"      # Override with PAM_DEFAULT_DEVICE=cuda on GPU deployments
    PAM_BASE_MODEL_PATH: str = "models_AL/pam/base/base_pam_model.pt"  # Physical base model file
    PAM_CHECKPOINTS_DIR: str = "models_AL/pam/checkpoints"              # Versioned checkpoint storage

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # Ignore env vars not in this model (e.g. ACTIVE_LEARNING_* from other branches)
    )


settings = Settings()
