"""
Celery application instance and configuration
"""

from celery import Celery
from app.config import settings
from app.logging_config import configure_logging

configure_logging()

# Create Celery instance
celery_app = Celery(
    "yapat",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.embedding_tasks",
        "app.tasks.processing_tasks",
        "app.tasks.pam_al_tasks",
    ]
)

# Configure Celery
celery_app.conf.update(
    task_track_started=settings.CELERY_TASK_TRACK_STARTED,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "app.tasks.embedding_tasks.*": {"queue": "embeddings"},
        "app.tasks.processing_tasks.*": {"queue": "processing"},
        "app.tasks.pam_al_tasks.*": {"queue": "pam_al"},
    },
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Result expiration
    result_expires=3600,  # Results expire after 1 hour
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    worker_hijack_root_logger=False,
    # Celery 6 deprecation: make startup broker retries explicit.
    broker_connection_retry_on_startup=True,
)

if __name__ == "__main__":
    celery_app.start()

