# YAPAT Backend

FastAPI backend application for YAPAT project.

## Prerequisites

- Docker and Docker Compose

## Quick Start with Docker Compose

The easiest way to run the entire application is using Docker Compose:

```bash
docker compose up
```

This single command will start:
- **PostgreSQL** database (port 5432)
- **Redis** for Celery (port 6379)
- **FastAPI** application (port 8000)
- **Celery worker** — general queues (embeddings, processing, exports, default)
- **Celery worker (PAM AL)** — dedicated worker for PAM Active Learning training tasks
- **Celery beat** scheduler for periodic tasks
- **Flower** monitoring dashboard (port 5555)

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json
- **Flower Dashboard**: http://localhost:5555

Database migrations run automatically on startup.

To run in detached mode (background):
```bash
docker compose up -d
```

To stop all services:
```bash
docker compose down
```

To stop and remove all data volumes:
```bash
docker compose down -v
```

## Updating from Latest Changes

If you're pulling updates from the repository (especially the pgvector migration):

1. **Pull latest code**
   ```bash
   git pull
   ```

2. **Stop running containers**
   ```bash
   docker-compose down
   ```

3. **Rebuild containers** (required for pgvector and new dependencies)
   ```bash
   docker-compose build --no-cache
   ```

4. **Start containers**
   ```bash
   docker-compose up -d
   ```

5. **Run database migrations**
   ```bash
   docker-compose exec api alembic upgrade head
   ```

6. **Verify migration**
   ```bash
   docker-compose exec api alembic current
   ```
   Should show: `2026_01_12_pgvector (head)`

7. **Test the API**
   ```bash
   curl http://localhost:8000/docs
   ```

**Important**: The `--no-cache` flag in step 3 ensures the pgvector PostgreSQL extension and Python package are properly installed.

## Manual Setup (Development)

If you prefer to run services manually:

1. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **Start services (PostgreSQL & Redis)**
   ```bash
   docker compose up -d db redis
   ```
   This starts:
   - PostgreSQL database (port 5432)
   - Redis for Celery (port 6379)

4. **Configure environment variables** (optional)
   
   Create a `.env` file in the root directory:
   ```env
   DATABASE_URL=postgresql://yapat_user:yapat_password@localhost/yapat
   SECRET_KEY=your-secret-key-change-in-production
   ```
   
   If not provided, defaults from `app/config.py` will be used.

   **Species model weights (WSSED / Active Learning):**  
   To use pre-trained species models, set `ACTIVE_LEARNING_MODELS_DIR` to the base directory that contains your `.pt` weights. Create one subdirectory per species (use lowercase, underscores for spaces), and place the checkpoint file inside it. The backend looks for filenames like `best_macro_model_segment.pt` or `best_micro_model.pt` (see `app/services/species_model_store.py`). Example:

   ```text
   models_AL/                    # ACTIVE_LEARNING_MODELS_DIR (e.g. ./models_AL or /path/to/models_AL)
   └── my_species_name/
       └── best_macro_model_segment.pt
   ```

   Add to `.env` (optional):

   ```env
   ACTIVE_LEARNING_MODELS_DIR=models_AL
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Start the general Celery worker** (in a separate terminal):
   ```bash
   ./start_celery_worker.sh
   ```

8. **Start the PAM Active Learning Celery worker** (in another separate terminal):
   ```bash
   celery -A app.celery_app worker \
       --loglevel=info \
       --concurrency=1 \
       --queues=pam_al \
       -n pam_al@%h
   ```
   > Concurrency is set to 1 because training loads large embedding matrices and
   > PyTorch models into memory. Set `PAM_DEFAULT_DEVICE=cuda` in your `.env`
   > to run on GPU.

9. **(Optional) Start Flower for monitoring:**
   ```bash
   ./start_flower.sh
   ```

## Celery Tasks

YAPAT uses Celery for asynchronous task processing across multiple queues:

| Queue | Worker flag | Purpose |
|-------|-------------|---------|
| `embeddings` | `--queues=embeddings` | BirdNET audio embedding generation |
| `processing` | `--queues=processing` | Dataset scanning, snippet creation |
| `exports` | `--queues=exports` | Annotation export / report generation |
| `default` | `--queues=default` | General fallback tasks |
| `pam_al` | `--queues=pam_al` | **PAM Active Learning** — train-from-scratch, manual retrain, auto-retrain from feedback |

### PAM Active Learning tasks

Heavy training operations are dispatched as background Celery tasks on the `pam_al` queue.
The API returns an `ALJobDispatch` response immediately with a `job_id`.
Poll `GET /api/pam-al/retrain/jobs/{job_id}` to track progress.

| Celery task | Triggered by |
|-------------|--------------|
| `pam_al_train_from_scratch` | `POST /api/pam-al/train-from-scratch` |
| `pam_al_manual_retrain` | `POST /api/pam-al/retrain/manual` |
| `pam_al_auto_retrain` | `POST /api/pam-al/feedback` (auto, when threshold is reached) |

**API Endpoints:**
All task endpoints are under `/api/tasks` — see interactive docs at http://localhost:8000/docs


## Database Management

- **Create a new migration**: `alembic revision --autogenerate -m "description"`
- **Apply migrations**: `alembic upgrade head`
- **Rollback**: `alembic downgrade -1`
- **Check current version**: `alembic current`

## Stop Services

Stop all services (PostgreSQL, Redis):
```bash
docker compose down
```

To also remove data volumes:
```bash
docker compose down -v
```
