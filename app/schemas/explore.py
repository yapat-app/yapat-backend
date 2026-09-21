"""Request schemas for the server-side explore API (/api/explore)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from app.services.explore.query import Filters

ScoreKey = Literal["uncertainty", "diversity", "density", "composite", "confidence"]
SortFieldName = Literal["confidence", "composite", "uncertainty", "diversity", "density", "date", "time"]
ProjectionMethodName = Literal["pca", "umap", "tsne", "isomap"]


class ExploreScope(BaseModel):
    dataset_id: int
    snippet_set_id: int
    checkpoint_id: Optional[int] = Field(
        default=None, description="Model checkpoint whose predictions drive scores; null = no model."
    )
    embedding_model_id: Optional[int] = Field(
        default=None, description="Required for projection endpoints; defaults to the snippet set's model."
    )


def _norm_strings(values: List[str]) -> List[str]:
    return sorted({v.strip() for v in values if isinstance(v, str) and v.strip()})


def _norm_range(value: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    lo, hi = float(value[0]), float(value[1])
    return (lo, hi) if lo <= hi else (hi, lo)


class ExploreFilters(BaseModel):
    annotation_status: Literal["any", "annotated", "unannotated"] = "any"
    annotated_species: List[str] = Field(default_factory=list, max_length=1000)
    predicted_species: List[str] = Field(default_factory=list, max_length=1000)
    label_scope: List[str] = Field(default_factory=list, max_length=1000)
    locations: List[str] = Field(default_factory=list, max_length=5000)
    date_range: Optional[Tuple[float, float]] = None
    months: List[int] = Field(default_factory=list, max_length=12)
    time_range: Optional[Tuple[float, float]] = None
    score_ranges: Dict[ScoreKey, Tuple[float, float]] = Field(default_factory=dict)
    sticky_ids: List[int] = Field(
        default_factory=list,
        max_length=5000,
        description="Snippets that stay admitted by the annotation-status filter (just-labelled rows).",
    )

    @field_validator("months")
    @classmethod
    def _months_in_range(cls, value: List[int]) -> List[int]:
        for month in value:
            if not 1 <= month <= 12:
                raise ValueError("months must be between 1 and 12")
        return value

    def to_query(self) -> Filters:
        score_ranges = []
        for key in sorted(self.score_ranges):
            lo, hi = self.score_ranges[key]
            lo, hi = max(0.0, min(1.0, float(lo))), max(0.0, min(1.0, float(hi)))
            if lo > hi:
                lo, hi = hi, lo
            score_ranges.append((key, lo, hi))
        return Filters(
            annotation_status=self.annotation_status,
            annotated_species=tuple(_norm_strings(self.annotated_species)),
            predicted_species=tuple(_norm_strings(self.predicted_species)),
            label_scope=tuple(_norm_strings(self.label_scope)),
            locations=tuple(sorted(set(self.locations))),
            date_range=_norm_range(self.date_range),
            months=tuple(sorted(set(self.months))),
            time_range=_norm_range(self.time_range),
            score_ranges=tuple(score_ranges),
            sticky_ids=tuple(sorted(set(int(i) for i in self.sticky_ids))),
        )


class ExploreSortField(BaseModel):
    field: SortFieldName
    direction: Literal["asc", "desc"] = "desc"


class SummaryRequest(BaseModel):
    scope: ExploreScope
    filters: ExploreFilters = Field(default_factory=ExploreFilters)
    bins: int = Field(default=28, ge=4, le=200)


class FacetsRequest(BaseModel):
    scope: ExploreScope


class FeedRequest(BaseModel):
    scope: ExploreScope
    filters: ExploreFilters = Field(default_factory=ExploreFilters)
    sort: List[ExploreSortField] = Field(default_factory=list, max_length=7)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)
    anchor_snippet_id: Optional[int] = Field(
        default=None,
        description="When set, the page containing this snippet is returned and its index reported.",
    )
    prefer_unlabeled: bool = Field(
        default=False,
        description="With anchor_snippet_id: resolve to the first unlabeled snippet at or after the anchor.",
    )


class RowsRequest(BaseModel):
    scope: ExploreScope
    filters: ExploreFilters = Field(default_factory=ExploreFilters)
    snippet_ids: List[int] = Field(default_factory=list, max_length=500)


class ProjectionRequest(BaseModel):
    scope: ExploreScope
    method: ProjectionMethodName = "pca"


class ProjectionStateRequest(BaseModel):
    scope: ExploreScope
    filters: ExploreFilters = Field(default_factory=ExploreFilters)
    method: ProjectionMethodName = "pca"
    pinned_ids: List[int] = Field(default_factory=list, max_length=2000)
    include_density: bool = True
    grid: int = Field(default=160, ge=16, le=512)


class ProjectionCoordsRequest(BaseModel):
    scope: ExploreScope
    method: ProjectionMethodName = "pca"
    snippet_ids: List[int] = Field(default_factory=list, max_length=2000)


class ViewportRequest(BaseModel):
    scope: ExploreScope
    filters: ExploreFilters = Field(default_factory=ExploreFilters)
    method: ProjectionMethodName = "pca"
    bbox: Tuple[float, float, float, float] = Field(description="xmin, xmax, ymin, ymax")
    max_points: int = Field(default=50_000, ge=1, le=200_000)
