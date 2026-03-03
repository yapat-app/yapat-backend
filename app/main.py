"""
FastAPI app entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import auth, teams, datasets, recordings, snippets, annotations, feed, invitations, tasks, taxonomy, embeddings, custom_taxonomy, pam_active_learning

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
)

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


@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "YAPAT Backend API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

