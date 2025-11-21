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

3. **Start PostgreSQL database**
   ```bash
   docker-compose up -d
   ```

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

Start the development server:
```bash
uvicorn app.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json


## Database Management

- **Create a new migration**: `alembic revision --autogenerate -m "description"`
- **Apply migrations**: `alembic upgrade head`
- **Rollback**: `alembic downgrade -1`
- **Check current version**: `alembic current`

## Stop the Database

```bash
docker-compose down
```

To also remove the data volume:
```bash
docker-compose down -v
```
