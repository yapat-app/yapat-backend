"""
Dataset endpoints
"""

from typing import List, Optional, Literal
from datetime import datetime
from io import StringIO
import csv

from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from app.api.deps import get_db, get_current_active_user, get_current_admin_user
from app.models.user import User, UserRole
from app.models.annotation import Annotation as AnnotationModel
from app.models.pam_active_learning import ALSnippetAnnotation, ALAnnotationSource
from app.models.snippet import Snippet
from app.models.recording import Recording
from app.models.embedding import SnippetSet, SnippetSetStatus
from app.schemas.dataset import (
    Dataset,
    DatasetCreate,
    DatasetUpdate,
    DatasetCreationResponse,
    DatasetExplorerResponse,
    AvailableDatasetPath,
    AvailableDatasetPathsResponse,
    SpeciesFolder,
    AudioFile
)
from app.schemas.annotation import AnnotationExport
from app.services.dataset_service import DatasetService
from app.utils.dataset_response import dataset_to_dict
from app.tasks.processing_tasks import process_dataset

router = APIRouter()


@router.post("/", response_model=DatasetCreationResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
        dataset_in: DatasetCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)

    if current_user.role != UserRole.ADMIN and dataset_in.team_id is None:
        raise HTTPException(status_code=400, detail="team_id is required for non-admin users")

    try:
        dataset = svc.create_dataset(dataset_in, current_user)
    except ValueError as e:
        if str(e) == "duplicate_dataset":
            raise HTTPException(status_code=409, detail="Dataset already exists")
        if str(e) == "team_not_found":
            raise HTTPException(status_code=404, detail="Team not found")
        if str(e) == "invalid_source_uri":
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid dataset path: {dataset_in.source_uri} does not exist or is not a directory"
            )
        raise

    # Dispatch background task for dataset processing (scanning + snippet generation)
    # Returns task ID for client tracking; None if task dispatch fails (backward compatible)
    try:
        task = process_dataset.delay(dataset.id)
        task_id = task.id
    except Exception:
        task_id = None

    dataset_response = Dataset(
        **dataset_to_dict(dataset, recording_count=0, is_ready_for_feed=False)
    )

    return DatasetCreationResponse(
        dataset=dataset_response,
        process_task_id=task_id,
        snippet_config_id=None,
        embedding_job_id=None,
    )


@router.get("/available-paths", response_model=AvailableDatasetPathsResponse)
def list_available_dataset_paths(
        prefix: Optional[str] = Query(
            None,
            description="Browse path relative to DATA_ROOT (e.g. ChorusRF). Empty = root.",
        ),
        db: Session = Depends(get_db),
        _admin: User = Depends(get_current_admin_user),
):
    """
    List child directories on the mounted data volume (DATA_ROOT) for dataset registration.
    Pass ``prefix`` to browse nested folders (e.g. ChorusRF → PrioritySpecies). Admin only.
    """
    svc = DatasetService(db)
    result = svc.list_available_source_paths(prefix=prefix)
    return AvailableDatasetPathsResponse(
        data_root=result["data_root"],
        current_path=result["current_path"],
        parent_path=result["parent_path"],
        paths=[AvailableDatasetPath(**p) for p in result["paths"]],
    )


@router.get("/", response_model=List[Dataset])
def read_datasets(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)
    datasets = svc.list_datasets(current_user=current_user, skip=skip, limit=limit)
    
    # Add recording count for each dataset
    dataset_ids = [ds.id for ds in datasets]
    if dataset_ids:
        recording_counts = (
            db.query(Recording.dataset_id, func.count(Recording.id).label('count'))
            .filter(Recording.dataset_id.in_(dataset_ids))
            .group_by(Recording.dataset_id)
            .all()
        )
        count_map = {ds_id: count for ds_id, count in recording_counts}
    else:
        count_map = {}
    
    # Check feed readiness for datasets with default snippet sets
    snippet_set_ids = [ds.default_snippet_set_id for ds in datasets if ds.default_snippet_set_id]
    if snippet_set_ids:
        ready_snippet_sets = (
            db.query(SnippetSet.id)
            .filter(
                SnippetSet.id.in_(snippet_set_ids),
                SnippetSet.status == SnippetSetStatus.READY
            )
            .all()
        )
        ready_set_ids = {ss_id for (ss_id,) in ready_snippet_sets}
    else:
        ready_set_ids = set()
    
    # Convert to schema and add recording_count and feed readiness
    result = []
    for dataset in datasets:
        is_ready = (
            dataset.default_snippet_set_id is not None 
            and dataset.default_snippet_set_id in ready_set_ids
        )
        
        result.append(
            Dataset(
                **dataset_to_dict(
                    dataset,
                    recording_count=count_map.get(dataset.id, 0),
                    is_ready_for_feed=is_ready,
                )
            )
        )
    
    return result


@router.get("/{dataset_id}", response_model=Dataset)
def read_dataset(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)
    dataset = svc.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Add recording count
    recording_count = (
        db.query(func.count(Recording.id))
        .filter(Recording.dataset_id == dataset_id)
        .scalar()
    ) or 0
    
    # Check if dataset is ready for feed generation
    is_ready_for_feed = False
    if dataset.default_snippet_set_id:
        snippet_set = (
            db.query(SnippetSet)
            .filter(SnippetSet.id == dataset.default_snippet_set_id)
            .first()
        )
        if snippet_set and snippet_set.status == SnippetSetStatus.READY:
            is_ready_for_feed = True
    
    return Dataset(
        **dataset_to_dict(
            dataset,
            recording_count=recording_count,
            is_ready_for_feed=is_ready_for_feed,
        )
    )


@router.patch("/{dataset_id}", response_model=Dataset)
def update_dataset(
    dataset_id: int,
    update_in: DatasetUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update dataset metadata. Admin or team owner for the dataset's team."""
    svc = DatasetService(db)
    try:
        dataset = svc.update_dataset(dataset_id, update_in, current_user)
    except ValueError as e:
        code = str(e)
        if code == "not_found":
            raise HTTPException(status_code=404, detail="Dataset not found")
        if code == "forbidden":
            raise HTTPException(status_code=403, detail="Not allowed to update this dataset")
        if code in (
            "invalid_spectrogram_f_min",
            "invalid_spectrogram_f_max",
            "spectrogram_f_max_lte_min",
            "invalid_source_uri",
        ):
            raise HTTPException(status_code=400, detail=code)
        raise

    recording_count = (
        db.query(func.count(Recording.id))
        .filter(Recording.dataset_id == dataset_id)
        .scalar()
    ) or 0

    is_ready_for_feed = False
    if dataset.default_snippet_set_id:
        snippet_set = (
            db.query(SnippetSet)
            .filter(SnippetSet.id == dataset.default_snippet_set_id)
            .first()
        )
        if snippet_set and snippet_set.status == SnippetSetStatus.READY:
            is_ready_for_feed = True

    return Dataset(
        **dataset_to_dict(
            dataset,
            recording_count=recording_count,
            is_ready_for_feed=is_ready_for_feed,
        )
    )


@router.get("/{dataset_id}/explorer", response_model=DatasetExplorerResponse)
def get_dataset_explorer(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    """
    Get dataset structure for explorer view.
    
    Returns species (subfolders) and their audio files by scanning
    the physical directory structure of the dataset.
    
    This endpoint is useful for:
    - Previewing dataset contents before processing
    - Exploring organized datasets with species in subfolders
    - Getting a quick overview of dataset organization
    """
    svc = DatasetService(db)
    dataset = svc.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        structure = svc.get_dataset_structure(dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan dataset structure: {str(e)}"
        )
    
    # Convert to response model
    species_list = []
    for species_data in structure['species']:
        files = [
            AudioFile(
                filename=f['filename'],
                file_path=f['file_path'],
                size=f['size']
            )
            for f in species_data['files']
        ]
        
        species_list.append(
            SpeciesFolder(
                name=species_data['name'],
                file_count=species_data['file_count'],
                files=files
            )
        )
    
    return DatasetExplorerResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        source_uri=dataset.source_uri,
        species=species_list
    )


@router.get("/{dataset_id}/recording-locations")
def get_recording_locations(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    """
    Return unique location values for all recordings in a dataset.

    Location is read from extra_metadata['location'] when present, otherwise
    parsed from the recording file name as the first segment before an underscore
    (e.g. 'SITE001' from 'SITE001_20240101_120000.wav').
    """
    recordings = (
        db.query(Recording.file_name, Recording.extra_metadata)
        .filter(Recording.dataset_id == dataset_id)
        .all()
    )

    locations: set[str] = set()
    for file_name, meta in recordings:
        if meta and isinstance(meta, dict) and meta.get("location"):
            locations.add(str(meta["location"]))
        elif file_name:
            stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
            part = stem.split("_")[0] if "_" in stem else stem
            if part:
                locations.add(part)

    return {"locations": sorted(locations)}


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)
    dataset = svc.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not svc.user_can_manage_dataset(current_user, dataset):
        raise HTTPException(status_code=403, detail="Not authorized to delete dataset")

    svc.delete_dataset(dataset)
    return None


EXPORT_CSV_HEADERS = [
    'annotation_id', 'dataset_id', 'snippet_id', 'taxon_id',
    'resolved_name_snapshot', 'in_scope', 'created_at', 'created_by',
    'recording_file_name', 'snippet_start_time',
    'snippet_end_time', 'snippet_duration',
]

# `source_table` is diagnostic — it exposes which of the two label stores a row
# came from, which is an internal detail annotators have no use for. Admins get
# it because it is what makes `annotation_id` unambiguous across the two tables.
EXPORT_ADMIN_ONLY_FIELDS = ['source_table']


def _resolve_scope_snippet_ids(
        db: Session,
        dataset_id: int,
        labels: List[str],
) -> set:
    """
    Snippet ids in the dataset carrying at least one of `labels`.

    A label may be written three ways depending on which store it landed in: an
    AL label string, a canonical namespaced taxon_id (`gbif:123`, `custom:uuid`)
    or a canonical resolved name. Accept any of them, so callers can scope by
    whatever spelling they have.
    """
    al_snippets = (
        db.query(ALSnippetAnnotation.snippet_id)
        .join(Snippet, ALSnippetAnnotation.snippet_id == Snippet.id)
        .join(Recording, Snippet.recording_id == Recording.id)
        .filter(
            Recording.dataset_id == dataset_id,
            ALSnippetAnnotation.source == ALAnnotationSource.USER,
            ALSnippetAnnotation.label.in_(labels),
        )
    )
    canonical_snippets = (
        db.query(AnnotationModel.snippet_id)
        .join(Snippet, AnnotationModel.snippet_id == Snippet.id)
        .join(Recording, Snippet.recording_id == Recording.id)
        .filter(
            Recording.dataset_id == dataset_id,
            (AnnotationModel.taxon_id.in_(labels))
            | (AnnotationModel.resolved_name_snapshot.in_(labels)),
        )
    )
    return {row[0] for row in al_snippets.all()} | {row[0] for row in canonical_snippets.all()}


def _collect_export_rows(
        db: Session,
        dataset_id: int,
        taxon_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        user_id: Optional[int] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
) -> List[dict]:
    """
    Collect a dataset's USER annotations from both label stores.

    Annotations live in two places and neither is a superset of the other:
    `al_snippet_annotation` (written by AL-mode feedback, and by the annotate
    feed) and the canonical `annotations` table (written by the classic hub).
    Mirroring between them only started in June 2026, so an export from either
    one alone silently drops history. This unions both, the same way the feed's
    filters do.

    Rows are deduplicated on (snippet_id, label, user) because the mirroring
    means most rows are present in both stores. The canonical row wins that tie:
    it carries a real taxon_id, where the AL store only keeps a single label
    string doing duty as both. Deduplication also collapses AL rows that differ
    only by model_checkpoint_id, which the AL-only export emitted more than once.

    Every row carries `source_table`; callers decide whether to expose it.

    `labels` (or its single-value alias `taxon_id`) scopes the export at snippet
    level: a snippet matching any requested label contributes *all* of its
    annotations, with `in_scope` marking the rows that matched. Around one
    snippet in ten here carries a species label alongside a recording condition
    (wind, rain, stream, artefact), and a row-level filter would drop that
    context. `user_id` and the date filters narrow the emitted rows only —
    scope is resolved from labels alone, so "snippets labelled X, showing only
    user Y's annotations" does not collapse into "snippets where Y applied X".
    """
    scope_labels = [label for label in (labels or []) if label]
    if taxon_id:
        scope_labels.append(taxon_id)

    scope_snippet_ids = (
        _resolve_scope_snippet_ids(db, dataset_id, scope_labels)
        if scope_labels else None
    )
    if scope_snippet_ids is not None and not scope_snippet_ids:
        return []

    al_query = (
        db.query(
            ALSnippetAnnotation.id.label('annotation_id'),
            Recording.dataset_id.label('dataset_id'),
            ALSnippetAnnotation.snippet_id,
            ALSnippetAnnotation.label.label('taxon_id'),
            ALSnippetAnnotation.label.label('resolved_name_snapshot'),
            ALSnippetAnnotation.created_at,
            ALSnippetAnnotation.user_id.label('created_by'),
            Recording.file_name.label('recording_file_name'),
            Snippet.start_time.label('snippet_start_time'),
            Snippet.end_time.label('snippet_end_time'),
            Snippet.duration.label('snippet_duration'),
        )
        .join(Snippet, ALSnippetAnnotation.snippet_id == Snippet.id)
        .join(Recording, Snippet.recording_id == Recording.id)
        .filter(Recording.dataset_id == dataset_id)
        .filter(ALSnippetAnnotation.source == ALAnnotationSource.USER)
    )
    if scope_snippet_ids is not None:
        al_query = al_query.filter(ALSnippetAnnotation.snippet_id.in_(scope_snippet_ids))
    if user_id:
        al_query = al_query.filter(ALSnippetAnnotation.user_id == user_id)
    if created_after:
        al_query = al_query.filter(ALSnippetAnnotation.created_at >= created_after)
    if created_before:
        al_query = al_query.filter(ALSnippetAnnotation.created_at <= created_before)

    canonical_query = (
        db.query(
            AnnotationModel.id.label('annotation_id'),
            Recording.dataset_id.label('dataset_id'),
            AnnotationModel.snippet_id,
            AnnotationModel.taxon_id,
            AnnotationModel.resolved_name_snapshot,
            AnnotationModel.created_at,
            AnnotationModel.user_id.label('created_by'),
            Recording.file_name.label('recording_file_name'),
            Snippet.start_time.label('snippet_start_time'),
            Snippet.end_time.label('snippet_end_time'),
            Snippet.duration.label('snippet_duration'),
        )
        .join(Snippet, AnnotationModel.snippet_id == Snippet.id)
        .join(Recording, Snippet.recording_id == Recording.id)
        .filter(Recording.dataset_id == dataset_id)
    )
    if scope_snippet_ids is not None:
        canonical_query = canonical_query.filter(
            AnnotationModel.snippet_id.in_(scope_snippet_ids)
        )
    if user_id:
        canonical_query = canonical_query.filter(AnnotationModel.user_id == user_id)
    if created_after:
        canonical_query = canonical_query.filter(AnnotationModel.created_at >= created_after)
    if created_before:
        canonical_query = canonical_query.filter(AnnotationModel.created_at <= created_before)

    scope_set = set(scope_labels)

    # The two stores have independent id sequences, so a bare row id collides
    # across them -- annotations 6..23 and al_snippet_annotation 6..23 are
    # unrelated rows. Exports were emitting the same annotation_id twice for
    # different labels, and `source_table` (the only disambiguator) is stripped
    # for non-admins. Prefixing makes the id unique for every reader.
    ID_PREFIXES = {'annotations': 'ann', 'al_snippet_annotation': 'al'}

    def _row_to_dict(row, source_table: str) -> dict:
        # With no scope every row is trivially in scope, which keeps the column
        # — and so the CSV shape — the same for every export.
        in_scope = not scope_set or bool(
            {row.taxon_id, row.resolved_name_snapshot} & scope_set
        )
        return {
            'annotation_id': f"{ID_PREFIXES[source_table]}:{row.annotation_id}",
            'dataset_id': row.dataset_id,
            'snippet_id': row.snippet_id,
            'taxon_id': row.taxon_id,
            'resolved_name_snapshot': row.resolved_name_snapshot,
            'in_scope': in_scope,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'created_by': row.created_by,
            'recording_file_name': row.recording_file_name,
            'snippet_start_time': row.snippet_start_time,
            'snippet_end_time': row.snippet_end_time,
            'snippet_duration': row.snippet_duration,
            'source_table': source_table,
        }

    merged: dict[tuple, dict] = {}
    for row in al_query.all():
        key = (row.snippet_id, row.resolved_name_snapshot, row.created_by)
        merged.setdefault(key, _row_to_dict(row, 'al_snippet_annotation'))
    for row in canonical_query.all():
        key = (row.snippet_id, row.resolved_name_snapshot, row.created_by)
        merged[key] = _row_to_dict(row, 'annotations')

    return sorted(
        merged.values(),
        key=lambda item: (item['snippet_id'], item['resolved_name_snapshot'] or ''),
    )


@router.get("/{dataset_id}/annotations/export")
def export_dataset_annotations(
        dataset_id: int,
        format: Literal["json", "csv"] = Query("json", description="Export format: json or csv"),
        taxon_id: Optional[str] = Query(None, description="Single-label scope; alias for `labels`"),
        labels: Optional[str] = Query(
            None,
            description=(
                "Comma-separated label scope. Snippets matching any of these "
                "labels are exported in full, including their other labels "
                "(marked in_scope=false). Matches an AL label, a canonical "
                "taxon_id, or a resolved species name."
            ),
        ),
        user_id: Optional[int] = Query(None, description="Filter by user_id (created_by)"),
        created_after: Optional[datetime] = Query(None, description="Filter annotations created after this datetime"),
        created_before: Optional[datetime] = Query(None, description="Filter annotations created before this datetime"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    """
    Export all annotations for a dataset with recording and snippet metadata.

    Unions the AL label store and the canonical annotations table (USER labels
    only; ground-truth imports are excluded), deduplicated per
    (snippet, label, user). See `_collect_export_rows`.

    Supports filtering by:
    - labels: Comma-separated label scope, resolved at snippet level — a snippet
      matching any of them is exported with all of its annotations, `in_scope`
      marking which rows matched. Preserves co-occurring context (a species
      alongside wind/rain/stream), which a row-level filter would drop.
    - taxon_id: Single-label alias for `labels`; supplying both is rejected
    - user_id: Filter by annotation creator
    - user_id: Filter by annotation creator
    - created_after: Filter annotations created after datetime
    - created_before: Filter annotations created before datetime

    Returns either JSON (default) or CSV format. Admins additionally get a
    `source_table` column; see EXPORT_ADMIN_ONLY_FIELDS.
    """
    # Verify dataset exists
    svc = DatasetService(db)
    dataset = svc.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if taxon_id and labels:
        raise HTTPException(
            status_code=400,
            detail="Provide either `labels` or `taxon_id`, not both.",
        )

    scope_labels = [part.strip() for part in labels.split(",")] if labels else []
    scope_labels = [part for part in scope_labels if part]

    annotations_data = _collect_export_rows(
        db,
        dataset_id,
        taxon_id=taxon_id,
        labels=scope_labels,
        user_id=user_id,
        created_after=created_after,
        created_before=created_before,
    )

    is_admin = current_user.role == UserRole.ADMIN
    if not is_admin:
        for row in annotations_data:
            for field in EXPORT_ADMIN_ONLY_FIELDS:
                row.pop(field, None)

    headers = EXPORT_CSV_HEADERS + (EXPORT_ADMIN_ONLY_FIELDS if is_admin else [])

    # Return based on format
    if format == "csv":
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(annotations_data)
        csv_content = output.getvalue()

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=dataset_{dataset_id}_annotations.csv"
            }
        )
    else:
        # Validate through the schema, then drop unset keys so non-admins get no
        # `source_table` at all rather than an explicit null.
        return [
            AnnotationExport(**data).model_dump(exclude_none=True)
            for data in annotations_data
        ]


# ── Quick Labels ────────────────────────────────────────────────────────────

@router.get("/{dataset_id}/quick-labels")
def get_quick_labels(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return dataset quick labels. Falls back to active checkpoint species list if not configured."""
    from app.models.dataset import Dataset as DatasetModel
    from app.models.pam_active_learning import ALModelFamilyState, ALModelCheckpoint
    from app.services.pam_al._checkpoint_helpers import list_active_family_checkpoints, load_species_from_label_config

    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.quick_labels is not None:
        return dataset.quick_labels

    # Fallback: pre-populate from active checkpoint species list (read-only, not persisted)
    checkpoints = list_active_family_checkpoints(db, dataset_id=dataset_id)
    for ckpt in checkpoints:
        if ckpt.label_config_path:
            try:
                species = load_species_from_label_config(ckpt.label_config_path)
                return [
                    {
                        "taxon_id": f"local:{s.lower().replace(' ', '_')[:120]}",
                        "display_name": s,
                    }
                    for s in species
                ]
            except Exception:
                continue
    return []


@router.put("/{dataset_id}/quick-labels")
def put_quick_labels(
    dataset_id: int,
    body: List[dict],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Replace the dataset's quick label list."""
    from app.models.dataset import Dataset as DatasetModel

    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    cleaned = [
        {"taxon_id": str(item["taxon_id"]), "display_name": str(item["display_name"])}
        for item in body
        if isinstance(item, dict) and item.get("taxon_id") and item.get("display_name")
    ]
    dataset.quick_labels = cleaned
    db.commit()
    return cleaned
