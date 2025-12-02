# YAPAT Backend

FastAPI backend application for YAPAT project.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- pip

## Setup

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
   docker-compose up -d
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

6. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

## Running the Application

### 1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json

### 2. Start Celery worker (for async tasks):
```bash
./start_celery_worker.sh
```

### 3. (Optional) Start Flower for monitoring:
```bash
./start_flower.sh
```
- **Monitoring Dashboard**: http://localhost:5555 (admin/yapat123)

## Celery Tasks

YAPAT uses Celery for asynchronous task processing:
- **Embedding generation**: Generate audio embeddings for snippets
- **Recording processing**: Process audio files and create snippets
- **Data export**: Export annotations and generate reports


**API Endpoints:**
All task endpoints are under `/api/tasks` - see interactive docs at http://localhost:8000/docs


## Database Management

- **Create a new migration**: `alembic revision --autogenerate -m "description"`
- **Apply migrations**: `alembic upgrade head`
- **Rollback**: `alembic downgrade -1`
- **Check current version**: `alembic current`

## Stop Services

Stop all services (PostgreSQL, Redis):
```bash
docker-compose down
```

To also remove data volumes:
```bash
docker-compose down -v
```
