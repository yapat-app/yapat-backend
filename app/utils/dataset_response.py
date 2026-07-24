"""Shared Dataset API response builder."""

from typing import Any, Optional

from app.models.dataset import Dataset as DatasetModel


def dataset_to_dict(
    dataset: DatasetModel,
    *,
    recording_count: int = 0,
    is_ready_for_feed: bool = False,
) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "source_uri": dataset.source_uri,
        "team_id": dataset.team_id,
        "dataset_type": dataset.dataset_type,
        "default_snippet_set_id": dataset.default_snippet_set_id,
        "spectrogram_f_min_hz": dataset.spectrogram_f_min_hz,
        "spectrogram_f_max_hz": dataset.spectrogram_f_max_hz,
        "retrain_after_threshold": dataset.retrain_after_threshold,
        "is_reference": dataset.is_reference,
        "reference_metadata_path": dataset.reference_metadata_path,
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
        "recording_count": recording_count,
        "is_ready_for_feed": is_ready_for_feed,
    }
