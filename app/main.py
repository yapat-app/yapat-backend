"""
FastAPI app entry point
"""

import logging
import time
from contextlib import asynccontextmanager

from anyio import to_thread
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import settings
from app.logging_config import configure_logging

configure_logging()

from app.api import (
    annotations,
    auth,
    custom_taxonomy,
    datasets,
    embeddings,
    feed,
    invitations,
    pam_active_learning,
    recordings,
    snippets,
    tasks,
    taxonomy,
    teams,
    visualisations,
    wssed,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Cap the threadpool that runs sync `def` endpoints.

    Almost every endpoint here is sync, so each request occupies one AnyIO
    worker thread and holds one DB connection for its duration. AnyIO defaults
    to 40 threads per process while the pool is much smaller, so under a burst
    the surplus threads block in pool.connect() and fail at DB_POOL_TIMEOUT --
    which is how a page fan-out turned into 30s and 60s response times.
    Capping this to the pool size makes requests wait for a thread instead,
    which shows up as latency rather than errors.
    """
    if settings.API_THREADPOOL_LIMIT is not None:
        to_thread.current_default_thread_limiter().total_tokens = (
            settings.API_THREADPOOL_LIMIT
        )
        logging.getLogger("yapat").info(
            "threadpool limit set to %d (db pool %d+%d per process)",
            settings.API_THREADPOOL_LIMIT,
            settings.DB_POOL_SIZE,
            settings.DB_MAX_OVERFLOW,
        )
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_STR}/openapi.json" if settings.ENABLE_DOCS else None,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    swagger_ui_parameters={
        "persistAuthorization": True,
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate", "X-Channels"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

_request_logger = logging.getLogger("yapat.request")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    _request_logger.info(
        "%s %s status=%d duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_STR}/auth", tags=["auth"])
app.include_router(teams.router, prefix=f"{settings.API_STR}/teams", tags=["teams"])
app.include_router(datasets.router, prefix=f"{settings.API_STR}/datasets", tags=["datasets"])
app.include_router(recordings.router, prefix=f"{settings.API_STR}/recordings", tags=["recordings"])
app.include_router(snippets.router, prefix=f"{settings.API_STR}/snippets", tags=["snippets"])
app.include_router(annotations.router, prefix=f"{settings.API_STR}/annotations", tags=["annotations"])
app.include_router(feed.router, prefix=f"{settings.API_STR}/feed", tags=["feed"])
app.include_router(invitations.router, prefix=f"{settings.API_STR}/invitations", tags=["invitations"])
app.include_router(tasks.router, prefix=f"{settings.API_STR}/tasks", tags=["tasks"])
app.include_router(taxonomy.router, prefix=f"{settings.API_STR}/taxonomy", tags=["taxonomy"])
app.include_router(custom_taxonomy.router, prefix=f"{settings.API_STR}/taxonomy", tags=["custom-taxonomy"])
app.include_router(embeddings.router, prefix=f"{settings.API_STR}", tags=["embeddings"])
app.include_router(pam_active_learning.router, prefix=f"{settings.API_STR}/pam-al", tags=["pam-active-learning"])
app.include_router(visualisations.router, prefix=f"{settings.API_STR}/visualisations", tags=["visualisations"])
app.include_router(wssed.router, prefix=f"{settings.API_STR}/wssed", tags=["wssed"])


@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "YAPAT Backend API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

