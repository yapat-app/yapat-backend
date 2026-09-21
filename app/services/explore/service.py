"""
Explore request handling: scope validation, cached filter evaluation and
response shaping. Endpoints in app/api/explore.py are thin wrappers.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models.embedding import SnippetSet, SnippetSetStatus
from app.models.pam_active_learning import ALModelCheckpoint, ALPrediction
from app.models.snippet import Snippet
from app.models.user import User
from app.models.visualisation import FPVVis
from app.schemas.explore import (
    ExploreFilters,
    ExploreScope,
    FeedRequest,
    ProjectionCoordsRequest,
    ProjectionRequest,
    ProjectionStateRequest,
    RowsRequest,
    SummaryRequest,
    ViewportRequest,
)
from app.services.explore import registry
from app.services.explore.builders import PROJECTION_METHODS
from app.services.explore.query import (
    SCORE_COLUMNS,
    BaseData,
    FilterResult,
    Filters,
    LabelsData,
    ModelData,
    apply_filters,
    compute_non_score_mask,
    compute_population,
    date_time_counts,
    order_rows,
    resolve_anchor,
    score_histograms,
    unpack_labels,
)
from app.services.explore.sampling import density_grid

MAX_EXTRA_POINTS = 20_000


class ScopeError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def b64_bits(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(bool), bitorder="little").tobytes()).decode("ascii")


@dataclass
class Frame:
    scope: ExploreScope
    embedding_model_id: int
    base: BaseData
    model: ModelData | None
    labels: LabelsData

    @property
    def versions(self) -> dict[str, str | None]:
        return {
            "base": self.base.version,
            "model": self.model.version if self.model else None,
            "labels": self.labels.version,
        }

    def cache_key(self) -> tuple:
        return (self.base.version, self.model.version if self.model else None, self.labels.version)


class ExploreService:
    def __init__(self, db: Session, user: User):
        self.db = db
        self.user = user

    # ── scope ─────────────────────────────────────────────────────────────

    def _validate_scope(self, scope: ExploreScope) -> SnippetSet:
        from app.services.dataset_service import DatasetService

        datasets = DatasetService(self.db)
        if datasets.get_dataset(scope.dataset_id) is None:
            raise ScopeError(404, "Dataset not found")
        if not datasets.user_can_access_dataset(self.user, scope.dataset_id):
            raise ScopeError(403, "Not authorized to access this dataset")
        snippet_set = self.db.get(SnippetSet, scope.snippet_set_id)
        if snippet_set is None:
            raise ScopeError(404, f"Snippet set {scope.snippet_set_id} not found")
        if snippet_set.dataset_id != scope.dataset_id:
            raise ScopeError(
                400,
                f"snippet_set_id={scope.snippet_set_id} belongs to dataset_id={snippet_set.dataset_id}, "
                f"not dataset_id={scope.dataset_id}.",
            )
        if snippet_set.status != SnippetSetStatus.READY:
            raise ScopeError(409, f"Snippet set {scope.snippet_set_id} is not ready yet.")
        if scope.checkpoint_id is not None:
            checkpoint = self.db.get(ALModelCheckpoint, scope.checkpoint_id)
            if checkpoint is None:
                raise ScopeError(404, f"Checkpoint {scope.checkpoint_id} not found")
            if checkpoint.dataset_id != scope.dataset_id:
                raise ScopeError(400, f"Checkpoint {scope.checkpoint_id} does not belong to this dataset.")
        if (
            scope.embedding_model_id is not None
            and scope.embedding_model_id != snippet_set.embedding_model_id
        ):
            raise ScopeError(
                400,
                f"embedding_model_id={scope.embedding_model_id} does not match snippet set "
                f"{snippet_set.id} (embedding_model_id={snippet_set.embedding_model_id}).",
            )
        return snippet_set

    def frame(self, scope: ExploreScope) -> Frame:
        snippet_set = self._validate_scope(scope)
        # Release the read transaction before any potentially long build.
        self.db.commit()
        base = registry.get_base(self.db, scope.snippet_set_id)
        model = None
        if scope.checkpoint_id is not None:
            model = registry.get_model(
                self.db,
                base,
                scope.checkpoint_id,
                scope.snippet_set_id,
                predictions_exist=lambda: self._predictions_exist(scope),
            )
        labels = registry.get_labels(self.db, base, scope.dataset_id, scope.snippet_set_id)
        return Frame(
            scope=scope,
            embedding_model_id=int(snippet_set.embedding_model_id),
            base=base,
            model=model,
            labels=labels,
        )

    def _predictions_exist(self, scope: ExploreScope) -> bool:
        return (
            self.db.query(ALPrediction.id)
            .join(Snippet, Snippet.id == ALPrediction.snippet_id)
            .filter(
                ALPrediction.model_checkpoint_id == scope.checkpoint_id,
                Snippet.snippet_set_id == scope.snippet_set_id,
            )
            .first()
            is not None
        )

    # ── cached filter evaluation ──────────────────────────────────────────

    def evaluate(self, frame: Frame, filters: Filters) -> FilterResult:
        base_key = frame.cache_key()
        population = registry.population_cache.get_or_compute(
            ("pop", base_key[0], base_key[1], filters.population_key()),
            lambda: compute_population(frame.base, frame.model, filters),
        )
        non_score = registry.non_score_cache.get_or_compute(
            ("ns", base_key, filters.non_score_key()),
            lambda: compute_non_score_mask(frame.base, frame.model, frame.labels, population, filters),
        )
        return registry.filter_cache.get_or_compute(
            ("vis", base_key, filters.visible_key()),
            lambda: apply_filters(
                frame.base, frame.model, frame.labels, filters,
                population=population, non_score=non_score,
            ),
        )

    def ordering(self, frame: Frame, filters: Filters, sort: Sequence[tuple[str, str]]) -> np.ndarray:
        result = self.evaluate(frame, filters)
        return registry.order_cache.get_or_compute(
            ("order", frame.cache_key(), filters.visible_key(), tuple(sort)),
            lambda: order_rows(frame.base, frame.model, result, sort),
        )

    # ── endpoints ─────────────────────────────────────────────────────────

    def summary(self, req: SummaryRequest) -> dict:
        frame = self.frame(req.scope)
        filters = req.filters.to_query()
        result = self.evaluate(frame, filters)
        return {
            "status": "ready",
            "versions": frame.versions,
            "has_model": frame.model is not None,
            "counts": {
                "total_snippets": frame.base.n,
                "population": int(result.population.mask.sum()),
                "non_score": int(result.non_score.sum()),
                "visible": int(result.visible.sum()),
                "labeled": int(frame.labels.labeled.sum()),
            },
            "domains": {k: [v[0], v[1]] for k, v in result.population.domains.items()},
            "bins": req.bins,
            "histograms": score_histograms(frame.model, result, req.bins, frame.base.n),
        }

    def facets(self, scope: ExploreScope) -> dict:
        frame = self.frame(scope)
        date_time = registry.base_cache.get_or_compute(
            ("datetime", frame.base.version), lambda: date_time_counts(frame.base)
        )
        return {
            "status": "ready",
            "versions": frame.versions,
            "locations": list(frame.base.locations),
            "annotated_species": list(frame.labels.vocab),
            "label_order": list(frame.model.label_order) if frame.model else [],
            **date_time,
        }

    def feed(self, req: FeedRequest) -> dict:
        frame = self.frame(req.scope)
        filters = req.filters.to_query()
        sort = [(s.field, s.direction) for s in req.sort]
        order = self.ordering(frame, filters, sort)
        result = self.evaluate(frame, filters)
        total = int(order.shape[0])

        offset = req.offset
        anchor_index = None
        if req.anchor_snippet_id is not None:
            anchor_row = int(frame.base.index_of([req.anchor_snippet_id])[0])
            anchor_index = resolve_anchor(order, anchor_row, frame.labels, req.prefer_unlabeled)
            if anchor_index is not None:
                offset = (anchor_index // req.limit) * req.limit

        page = order[offset : offset + req.limit]
        return {
            "status": "ready",
            "versions": frame.versions,
            "total": total,
            "offset": offset,
            "limit": req.limit,
            "anchor_index": anchor_index,
            "anchor_snippet_id": (
                int(frame.base.snippet_ids[order[anchor_index]]) if anchor_index is not None else None
            ),
            "rows": self._rows(frame, result, page),
        }

    def rows(self, req: RowsRequest) -> dict:
        frame = self.frame(req.scope)
        filters = req.filters.to_query()
        result = self.evaluate(frame, filters)
        rows = frame.base.index_of(req.snippet_ids)
        rows = rows[rows >= 0]
        return {
            "status": "ready",
            "versions": frame.versions,
            "rows": self._rows(frame, result, rows),
            "visible": [bool(result.visible[r]) for r in rows],
        }

    def _rows(self, frame: Frame, result: FilterResult, rows: np.ndarray) -> list[dict]:
        base, model, labels = frame.base, frame.model, frame.labels
        out: list[dict] = []
        checkpoint_id = frame.scope.checkpoint_id
        for r in np.asarray(rows, dtype=np.int64):
            r = int(r)
            rec = int(base.recording_idx[r])
            row: dict[str, Any] = {
                "id": None,
                "model_checkpoint_id": checkpoint_id,
                "snippet_id": int(base.snippet_ids[r]),
                "recording_id": int(base.rec_ids[rec]) if rec >= 0 else None,
                "duration_sec": float(base.duration[r]),
                "predicted_labels": None,
                "predicted_label": "—",
                "confidence": 0.0,
                "uncertainty": None,
                "diversity": None,
                "density": None,
                "composite_score": None,
                "ranking_score": None,
                "created_at": None,
                "scores": {},
                "labels": labels.labels_for(r),
            }
            if model is not None and model.has_prediction[r]:
                mrow = int(model.model_row[r])
                predicted = unpack_labels(model.pred_bits[mrow], model.label_order)
                row["predicted_labels"] = predicted
                probs = np.asarray(model.probs[mrow], dtype=np.float32)
                best_label = predicted[0] if predicted else "—"
                if probs.size:
                    filled = np.where(np.isnan(probs), -np.inf, probs)
                    best = int(np.argmax(filled))
                    if filled[best] > 0:
                        best_label = model.label_order[best]
                row["predicted_label"] = best_label
                scores: dict[str, float] = {}
                for col, key in enumerate(SCORE_COLUMNS):
                    value = float(model.scores[r, col])
                    if math.isfinite(value):
                        scores[key] = value
                        row["composite_score" if key == "composite" else key] = value
                row["ranking_score"] = scores.get("composite")
                conf = float(result.population.confidence[r])
                if math.isfinite(conf) and conf > 0:
                    scores["confidence"] = conf
                    row["confidence"] = conf
                row["scores"] = scores
            out.append(row)
        return out

    # ── projection ────────────────────────────────────────────────────────

    def _projection_layer(self, frame: Frame):
        dataset_id = frame.scope.dataset_id
        em = frame.embedding_model_id

        def exists() -> bool:
            return (
                self.db.query(FPVVis.id)
                .filter(FPVVis.dataset_id == dataset_id)
                .filter(FPVVis.embedding_model_id == em)
                .filter(FPVVis.model_checkpoint_id.is_(None))
                .first()
                is not None
            )

        return registry.get_projection(self.db, dataset_id, em, exists)

    def _method_view(self, frame: Frame, layer, method: str) -> dict:
        """Static per (base, projection, method): sample rows + coordinate masks."""

        def compute() -> dict:
            maps = registry.get_projection_maps(frame.base, layer)
            coords = np.asarray(layer[f"coords_{method}"], dtype=np.float32)
            finite = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
            in_base = maps.proj_to_base >= 0
            has_coords_base = np.zeros(frame.base.n, dtype=bool)
            usable = finite & in_base
            has_coords_base[maps.proj_to_base[usable]] = True

            sample_proj = np.asarray(layer[f"sample_{method}"], dtype=np.int64)
            sample_proj = sample_proj[usable[sample_proj]] if sample_proj.size else sample_proj
            sample_base = maps.proj_to_base[sample_proj]
            in_sample = np.zeros(frame.base.n, dtype=bool)
            in_sample[sample_base] = True
            method_meta = (layer.meta.get("methods") or {}).get(method) or {}
            total_points = int(has_coords_base.sum())
            return {
                "coords": coords,
                "maps": maps,
                "has_coords_base": has_coords_base,
                "sample_proj": sample_proj,
                "sample_base": sample_base,
                "in_sample": in_sample,
                "bounds": method_meta.get("bounds"),
                "total_points": total_points,
                "sampled": total_points > sample_base.shape[0],
            }

        return registry.projection_map_cache.get_or_compute(
            ("view", frame.base.version, layer.version, method), compute
        )

    @staticmethod
    def _availability(method: str, n: int, finite: int) -> tuple[bool, str | None]:
        caps = {
            "umap": settings.FPV_UMAP_MAX_POINTS,
            "tsne": settings.FPV_TSNE_MAX_POINTS,
            "isomap": settings.FPV_ISOMAP_MAX_POINTS,
        }
        cap = caps.get(method)
        label = "t-SNE" if method == "tsne" else method.upper()
        if finite == 0:
            if cap is not None and n > cap:
                return False, f"Dataset has {n:,} points, exceeding the {cap:,}-point limit for {label}."
            return False, f"No valid {method} projection coordinates; try PCA or regenerate FPV."
        return True, None

    def projection(self, req: ProjectionRequest) -> dict:
        frame = self.frame(req.scope)
        layer = self._projection_layer(frame)
        method = req.method

        def compute() -> dict:
            view = self._method_view(frame, layer, method)
            available, reason = self._availability(
                method, int(layer.meta.get("n", 0)), view["total_points"]
            )
            coords = view["coords"][view["sample_proj"]]
            return {
                "status": "ready",
                "versions": {**frame.versions, "projection": layer.version},
                "method": method,
                "available": available,
                "reason": reason,
                "total_points": view["total_points"],
                "sampled": view["sampled"],
                "point_count": int(view["sample_base"].shape[0]),
                "bounds": view["bounds"],
                "ids": b64(frame.base.snippet_ids[view["sample_base"]], "<i4"),
                "x": b64(coords[:, 0], "<f4"),
                "y": b64(coords[:, 1], "<f4"),
            }

        payload = registry.projection_map_cache.get_or_compute(
            ("payload", frame.base.version, layer.version, method), compute
        )
        # Labels/model versions move independently of the static payload.
        return {**payload, "versions": {**frame.versions, "projection": layer.version}}

    def projection_state(self, req: ProjectionStateRequest) -> dict:
        frame = self.frame(req.scope)
        layer = self._projection_layer(frame)
        view = self._method_view(frame, layer, req.method)
        filters = req.filters.to_query()
        result = self.evaluate(frame, filters)
        labels = frame.labels
        label_dtype = "<i2" if len(labels.vocab) < 32767 else "<i4"

        sample_base = view["sample_base"]
        has_coords = view["has_coords_base"]
        visible_with_coords = result.visible & has_coords

        # Labeled or pinned points that the static sample doesn't include.
        pinned_rows = frame.base.index_of(req.pinned_ids)
        pinned_rows = pinned_rows[pinned_rows >= 0]
        candidates = labels.labeled & has_coords & ~view["in_sample"]
        pinned_mask = np.zeros(frame.base.n, dtype=bool)
        pinned_mask[pinned_rows] = True
        pinned_extra = np.flatnonzero(pinned_mask & has_coords & ~view["in_sample"])
        labeled_extra = np.flatnonzero(candidates & ~pinned_mask)
        budget = max(0, MAX_EXTRA_POINTS - pinned_extra.size)
        if labeled_extra.size > budget:
            priority = np.asarray(layer["priority"])[view["maps"].base_to_proj[labeled_extra]]
            labeled_extra = labeled_extra[np.argsort(priority, kind="stable")[:budget]]
        extra_rows = np.concatenate([pinned_extra[:MAX_EXTRA_POINTS], labeled_extra])
        extra_coords = view["coords"][view["maps"].base_to_proj[extra_rows]]

        response: dict[str, Any] = {
            "status": "ready",
            "versions": {**frame.versions, "projection": layer.version},
            "method": req.method,
            "total_points": view["total_points"],
            "visible_points": int(visible_with_coords.sum()),
            "point_count": int(sample_base.shape[0]),
            "visible": b64_bits(result.visible[sample_base]),
            "label_idx": b64(labels.first_label[sample_base], label_dtype),
            "label_dtype": label_dtype,
            "label_vocab": list(labels.vocab),
            "extras": {
                "count": int(extra_rows.shape[0]),
                "ids": b64(frame.base.snippet_ids[extra_rows], "<i4"),
                "x": b64(extra_coords[:, 0], "<f4"),
                "y": b64(extra_coords[:, 1], "<f4"),
                "visible": b64_bits(result.visible[extra_rows]),
                "label_idx": b64(labels.first_label[extra_rows], label_dtype),
            },
            "density": None,
        }

        if view["sampled"] and req.include_density and view["bounds"]:
            bounds = tuple(view["bounds"])

            def total_grid() -> np.ndarray:
                rows = np.flatnonzero(has_coords)
                c = view["coords"][view["maps"].base_to_proj[rows]]
                return density_grid(c[:, 0], c[:, 1], bounds, req.grid, req.grid)

            total = registry.projection_map_cache.get_or_compute(
                ("density", frame.base.version, layer.version, req.method, req.grid), total_grid
            )
            rows = np.flatnonzero(visible_with_coords)
            c = view["coords"][view["maps"].base_to_proj[rows]]
            visible_grid = density_grid(c[:, 0], c[:, 1], bounds, req.grid, req.grid)
            response["density"] = {
                "nx": req.grid,
                "ny": req.grid,
                "bounds": list(bounds),
                "total": b64(total, "<u4"),
                "visible": b64(visible_grid, "<u4"),
            }
        return response

    def projection_coords(self, req: ProjectionCoordsRequest) -> dict:
        """Coordinates for specific snippets (selection highlights outside the sample)."""
        frame = self.frame(req.scope)
        layer = self._projection_layer(frame)
        view = self._method_view(frame, layer, req.method)
        rows = frame.base.index_of(req.snippet_ids)
        rows = rows[rows >= 0]
        rows = rows[view["has_coords_base"][rows]]
        coords = view["coords"][view["maps"].base_to_proj[rows]]
        return {
            "status": "ready",
            "versions": {**frame.versions, "projection": layer.version},
            "method": req.method,
            "count": int(rows.shape[0]),
            "ids": b64(frame.base.snippet_ids[rows], "<i4"),
            "x": b64(coords[:, 0], "<f4"),
            "y": b64(coords[:, 1], "<f4"),
        }

    def viewport(self, req: ViewportRequest) -> dict:
        frame = self.frame(req.scope)
        layer = self._projection_layer(frame)
        view = self._method_view(frame, layer, req.method)
        filters = req.filters.to_query()
        result = self.evaluate(frame, filters)
        xmin, xmax, ymin, ymax = req.bbox
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        rows = np.flatnonzero(view["has_coords_base"])
        coords = view["coords"][view["maps"].base_to_proj[rows]]
        inside = (
            (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax)
            & (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
        )
        rows, coords = rows[inside], coords[inside]
        complete = rows.shape[0] <= req.max_points
        if not complete:
            priority = np.asarray(layer["priority"])[view["maps"].base_to_proj[rows]]
            keep = np.argpartition(priority, req.max_points - 1)[: req.max_points]
            keep.sort()
            rows, coords = rows[keep], coords[keep]
        labels = frame.labels
        label_dtype = "<i2" if len(labels.vocab) < 32767 else "<i4"
        return {
            "status": "ready",
            "versions": {**frame.versions, "projection": layer.version},
            "method": req.method,
            "bbox": [xmin, xmax, ymin, ymax],
            "complete": bool(complete),
            "count": int(rows.shape[0]),
            "ids": b64(frame.base.snippet_ids[rows], "<i4"),
            "x": b64(coords[:, 0], "<f4"),
            "y": b64(coords[:, 1], "<f4"),
            "visible": b64_bits(result.visible[rows]),
            "label_idx": b64(labels.first_label[rows], label_dtype),
            "label_dtype": label_dtype,
            "label_vocab": list(labels.vocab),
        }


def filters_from(model: ExploreFilters) -> Filters:
    return model.to_query()


__all__ = ["ExploreService", "ScopeError", "PROJECTION_METHODS"]
