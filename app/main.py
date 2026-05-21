"""
FastAPI app entry point
"""

import logging
import time

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

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_STR}/openapi.json",
    swagger_ui_parameters={
        "persistAuthorization": True,  # Keep authorization after page refresh
    }
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

