"""Schemas for the label sharing review queue."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ClipInfo(BaseModel):
    snippet_id: int
    snippet_set_id: int
    recording_id: int
    recording_file_name: Optional[str] = None
    start_time: float
    end_time: float


class LabelShareCandidateOut(BaseModel):
    id: int
    dataset_id: int
    status: str
    reason: str = Field(
        description=(
            "partial_overlap: the clips only partly overlap. target_shorter: a species "
            "label from a longer clip, the call may be outside this one. target_longer: "
            "a no-event label from a shorter clip, the rest of this one may have a call. "
            "conflict: this clip has a label of the opposite kind. source_removed: a "
            "recheck, the label this clip's reviewed label came from was deleted."
        )
    )
    taxon_id: str
    resolved_name_snapshot: str
    label: str = Field(description="The label text as the hub shows it.")
    overlap_seconds: float
    source_annotation_id: Optional[int] = Field(None, description="None for a recheck: the source was deleted.")
    target_annotation_id: Optional[int] = Field(None, description="A recheck: the label to keep or remove.")
    source_labeled_by: Optional[str] = None
    source: ClipInfo
    target: ClipInfo
    target_labels: list[str] = Field(default_factory=list, description="Labels the target clip has now.")
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    result_annotation_id: Optional[int] = None
    created_at: datetime


class LabelShareCandidateList(BaseModel):
    items: list[LabelShareCandidateOut]
    total: int
    skip: int
    limit: int


class LabelShareSummary(BaseModel):
    dataset_id: int
    auto_shared: int = Field(description="Labels copied automatically.")
    pending: int
    accepted: int
    rejected: int
    relabelled: int
    pending_by_snippet_set: dict[int, int] = Field(default_factory=dict)
    pending_by_snippet_set_reason: dict[int, dict[str, int]] = Field(
        default_factory=dict,
        description="Pending per snippet set, then per reason.",
    )
    labelled_by_snippet_set: dict[int, int] = Field(
        default_factory=dict,
        description="Clips with a label made by a person (not a copy), per snippet set.",
    )


class SnippetSetBreakdown(BaseModel):
    """One snippet set's snippets by label status; each snippet counts once."""

    snippet_set_id: int
    window_size: float
    status: str
    total_snippets: int
    recording_count: int = Field(description="Recordings with at least one snippet in this set.")
    labelled: int = Field(description="Snippets with a label made or accepted by a person.")
    auto_copied: int = Field(description="Snippets whose only labels are unreviewed automatic copies.")
    to_review: int = Field(description="Pending suggestions for this set's snippets.")


class SnippetBreakdown(BaseModel):
    dataset_id: int
    snippet_sets: list[SnippetSetBreakdown]


class ShareDatasetRequest(BaseModel):
    """A one-way transfer between two models' clips. Empty: share every way."""

    from_embedding_model_id: Optional[int] = None
    to_embedding_model_id: Optional[int] = None


class AcceptCandidateRequest(BaseModel):
    embedding_model_id: Optional[int] = Field(
        None, description="Embedding model selected while reviewing, recorded on the new label."
    )


class RelabelCandidateRequest(BaseModel):
    taxon_id: str = Field(..., pattern=r"^([a-z]+:[a-zA-Z0-9_-]+|custom:[a-f0-9-]+)$")
    display_name: Optional[str] = Field(None, description="Label text, e.g. the quick label name.")
    embedding_model_id: Optional[int] = None


class ReviewResult(BaseModel):
    candidate: LabelShareCandidateOut
    annotation_id: Optional[int] = None


class ShareDatasetResponse(BaseModel):
    dataset_id: int
    task_id: Optional[str] = None
