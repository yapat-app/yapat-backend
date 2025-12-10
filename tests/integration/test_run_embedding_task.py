import soundfile as sf
import numpy as np
from pathlib import Path

from app.tasks.embedding_tasks import run_embedding
from app.services.embedding_service import EmbeddingService
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.embedding import EmbeddingModel, EmbeddingJobStatus


def test_run_embedding_task_generates_snippets_and_embeddings(
    db_session,
    temp_data_root,
    tiny_wav_file,
):
    # ------------------------------------------
    # 1. Prepare dataset + recording
    # ------------------------------------------
    dataset = Dataset(
        name="TestDS",
        source_uri=str(temp_data_root),
    )
    db_session.add(dataset)
    db_session.commit()

    # Create a WAV inside dataset folder
    audio_path = temp_data_root / "rec1.wav"
    tiny_wav_file(audio_path, duration_sec=1.0, sr=16000)  # 1 second wav

    # Insert recording row
    rec = Recording(
        dataset_id=dataset.id,
        file_path=str(audio_path),
        duration_seconds=1.0,
    )
    db_session.add(rec)
    db_session.commit()

    # ------------------------------------------
    # 2. Create embedding model + job
    # ------------------------------------------
    model = EmbeddingModel(
        name="birdnet",
        version="1.0",
        default_window_size=0.5,
        default_step_size=0.5,
        default_overlap=0.0,
    )
    db_session.add(model)
    db_session.commit()

    service = EmbeddingService(db_session)
    job = service.create_embedding_job(dataset, model)

    # ------------------------------------------
    # 3. Run Celery embedding pipeline
    # ------------------------------------------
    result = run_embedding.delay(job.id).get()
    assert result["status"] == "completed"

    # ------------------------------------------
    # 4. Validate job state
    # ------------------------------------------
    job = service.get_job(job.id)
    assert job.status in (EmbeddingJobStatus.COMPLETED, EmbeddingJobStatus.FAILED)

    # ------------------------------------------
    # 5. Snippets generated
    # ------------------------------------------
    snippets = job.snippets
    assert len(snippets) > 0

    # With 1 second audio, window=0.5, step=0.5:
    # should yield exactly 2 snippets: [0–0.5], [0.5–1.0]
    times = [(s.start_time, s.end_time) for s in snippets]
    assert times == [(0.0, 0.5), (0.5, 1.0)]
