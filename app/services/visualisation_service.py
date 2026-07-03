import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.database import SessionLocal

from app.models.pam_active_learning import ALPrediction, ALModelFamilyState, ALModelCheckpoint
from app.schemas.pam_active_learning import ALModelType
from app.models.embedding import EmbeddingVector, SnippetSet
from app.models.snippet import Snippet
from app.models.visualisation import FPVVis
from app.schemas.visualisation import (
    FPVRequest,
    FPVDatasetRequest,
    FPVResponse,
    FPVPointMetadata,
    FPVProjection2D,
    FPVProjection3D,
    FPVVisibilityField,
    FPVVisibilityRangeResponse,
    FPVColorField,
    FPVColorMetadata,
    FPVMethod,
)
from utils.dr_methods import (
    build_knn_graph, pre_reduce_pca, run_dr_isomap, run_dr_tsne, run_dr_umap,
)

logger = logging.getLogger(__name__)

_FPV_COORD_COLUMNS = (
    "pca_2d_x", "pca_2d_y", "pca_3d_x", "pca_3d_y", "pca_3d_z",
    "umap_2d_x", "umap_2d_y", "umap_3d_x", "umap_3d_y", "umap_3d_z",
    "tsne_2d_x", "tsne_2d_y", "tsne_3d_x", "tsne_3d_y", "tsne_3d_z",
    "isomap_2d_x", "isomap_2d_y", "isomap_3d_x", "isomap_3d_y", "isomap_3d_z",
)


def _fpv_vis_row_values(
    dataset_id: int,
    embedding_model_id: int,
    snippet_id: int,
    index: int,
    coords: dict,
) -> dict:
    row: dict = {
        "dataset_id": dataset_id,
        "model_checkpoint_id": None,
        "embedding_model_id": embedding_model_id,
        "snippet_id": snippet_id,
    }
    if "pca_2d" in coords:
        row["pca_2d_x"] = float(coords["pca_2d"][index, 0])
        row["pca_2d_y"] = float(coords["pca_2d"][index, 1])
    if "pca_3d" in coords:
        row["pca_3d_x"] = float(coords["pca_3d"][index, 0])
        row["pca_3d_y"] = float(coords["pca_3d"][index, 1])
        row["pca_3d_z"] = float(coords["pca_3d"][index, 2])
    if "umap_2d" in coords:
        row["umap_2d_x"] = float(coords["umap_2d"][index, 0])
        row["umap_2d_y"] = float(coords["umap_2d"][index, 1])
    if "umap_3d" in coords:
        row["umap_3d_x"] = float(coords["umap_3d"][index, 0])
        row["umap_3d_y"] = float(coords["umap_3d"][index, 1])
        row["umap_3d_z"] = float(coords["umap_3d"][index, 2])
    if "tsne_2d" in coords:
        row["tsne_2d_x"] = float(coords["tsne_2d"][index, 0])
        row["tsne_2d_y"] = float(coords["tsne_2d"][index, 1])
    if "tsne_3d" in coords:
        row["tsne_3d_x"] = float(coords["tsne_3d"][index, 0])
        row["tsne_3d_y"] = float(coords["tsne_3d"][index, 1])
        row["tsne_3d_z"] = float(coords["tsne_3d"][index, 2])
    if "isomap_2d" in coords:
        row["isomap_2d_x"] = float(coords["isomap_2d"][index, 0])
        row["isomap_2d_y"] = float(coords["isomap_2d"][index, 1])
    if "isomap_3d" in coords:
        row["isomap_3d_x"] = float(coords["isomap_3d"][index, 0])
        row["isomap_3d_y"] = float(coords["isomap_3d"][index, 1])
        row["isomap_3d_z"] = float(coords["isomap_3d"][index, 2])
    return row


def persist_fpv_vis_dataset_rows(
    dataset_id: int,
    embedding_model_id: int,
    snippet_ids: list[int],
    coords: dict,
    chunk_size: int = 5000,
) -> None:
    """Write dataset-level FPV rows using a fresh DB session and chunked upserts."""
    total = len(snippet_ids)
    total_chunks = (total + chunk_size - 1) // chunk_size
    logger.info(
        "fpv dataset: saving %s rows for dataset_id=%s embedding_model_id=%s in %s chunks",
        total, dataset_id, embedding_model_id, total_chunks,
    )

    write_db = SessionLocal()
    try:
        bind = write_db.get_bind()
        use_pg_upsert = bind.dialect.name == "postgresql"

        if use_pg_upsert:
            for chunk_idx, start in enumerate(range(0, total, chunk_size), start=1):
                chunk_ids = snippet_ids[start : start + chunk_size]
                values = [
                    _fpv_vis_row_values(dataset_id, embedding_model_id, sid, start + i, coords)
                    for i, sid in enumerate(chunk_ids)
                ]
                stmt = pg_insert(FPVVis).values(values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["embedding_model_id", "snippet_id"],
                    index_where=FPVVis.model_checkpoint_id.is_(None),
                    set_={col: stmt.excluded[col] for col in _FPV_COORD_COLUMNS},
                )
                write_db.execute(stmt)
                write_db.commit()
                logger.info(
                    "fpv dataset: upsert chunk %s/%s (rows=%s)",
                    chunk_idx, total_chunks, len(chunk_ids),
                )
        else:
            VISService(write_db)._upsert_fpv_vis_dataset_rows(
                dataset_id, embedding_model_id, snippet_ids, coords,
            )
            write_db.commit()

    except Exception:
        write_db.rollback()
        raise
    finally:
        write_db.close()


class VISService:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_fpv(self, body: FPVRequest) -> FPVResponse:
        model_ckpt = self._get_active_checkpoint(
            dataset_id=body.dataset_id,
            model_family_name=body.model_family_name,
        )

        checkpoint_id = model_ckpt.id


        has_fpv = self._fpv_exists(checkpoint_id, body.run_3d)

        if not has_fpv:
            if model_ckpt.model_type == ALModelType.PAM_LINEAR_MULTILABEL.value:
                self._copy_dataset_fpv_to_checkpoint(
                    dataset_id=body.dataset_id,
                    model_ckpt=model_ckpt,
                    run_3d=body.run_3d,
                )
            else:
                self._generate_and_store_fpv(
                    dataset_id=body.dataset_id,
                    model_ckpt=model_ckpt,
                    run_3d=body.run_3d,
                )

        return self._build_fpv_response(body=body, checkpoint_id=checkpoint_id)

    def _fpv_exists(self, checkpoint_id: int, run_3d: bool) -> bool:
        row = (
            self.db.query(FPVVis)
            .filter(FPVVis.model_checkpoint_id == checkpoint_id)
            .first()
        )
        if row is None:
            return False

        if run_3d:
            return row.pca_3d_x is not None or row.umap_3d_x is not None or row.tsne_3d_x is not None or row.isomap_3d_x is not None

        return True

    def _generate_and_store_fpv(self, dataset_id: int, model_ckpt: ALModelCheckpoint, run_3d: bool) -> None:

        checkpoint_id = model_ckpt.id
        embedding_model_id = (model_ckpt.hyperparameters or {}).get("embedding_model_id")
        predictions = (
            self.db.query(ALPrediction)
            .filter(ALPrediction.model_checkpoint_id == checkpoint_id)
            .all()
        )

        if not predictions:
            raise ValueError(f"No predictions found for active checkpoint_id={checkpoint_id}.")

        rows_with_embeddings = [p for p in predictions if p.embedding is not None]
        if not rows_with_embeddings:
            raise ValueError(f"No prediction embeddings found for active checkpoint_id={checkpoint_id}.")

        snippet_ids = [p.snippet_id for p in rows_with_embeddings]
        X = np.array([p.embedding for p in rows_with_embeddings], dtype=np.float32)

        coords = self._compute_visualizations(X=X, run_3d=run_3d)

        self._upsert_fpv_vis_rows(
            dataset_id=dataset_id,
            model_checkpoint_id = checkpoint_id,
            embedding_model_id=embedding_model_id,
            snippet_ids=snippet_ids,
            coords=coords,
        )

        self.db.commit()

    def _build_fpv_response(self, body: FPVRequest, checkpoint_id: int) -> FPVResponse:

        rows = (
            self.db.query(FPVVis, ALPrediction)
            .join(
                ALPrediction,
                (ALPrediction.model_checkpoint_id == FPVVis.model_checkpoint_id)
                & (ALPrediction.snippet_id == FPVVis.snippet_id),
            )
            .filter(FPVVis.model_checkpoint_id == checkpoint_id)
            .order_by(FPVVis.snippet_id.asc())
            .all()
        )

        if not rows:
            raise ValueError(
                f"No feature projection rows found for active checkpoint_id={checkpoint_id}. "
                f"Generate projections first."
            )

        points = []
        projections_2d = {
            "pca": FPVProjection2D(x=[], y=[]),
            "umap": FPVProjection2D(x=[], y=[]),
            "tsne": FPVProjection2D(x=[], y=[]),
            "isomap": FPVProjection2D(x=[], y=[]),
        }

        projections_3d = {
            "pca": FPVProjection3D(x=[], y=[], z=[]),
            "umap": FPVProjection3D(x=[], y=[], z=[]),
            "tsne": FPVProjection3D(x=[], y=[], z=[]),
            "isomap": FPVProjection3D(x=[], y=[], z=[]),
        }

        has_any_3d = False
        color_values = []
        color_mode = "none"

        for vis_row, pred_row in rows:
            if not self._passes_visibility_filter(pred_row, body):
                continue
            color_value, color_mode = self._build_color_values(pred_row, body.color_filter_value)
            color_values.append(color_value)
            points.append(
                FPVPointMetadata(
                    snippet_id=pred_row.snippet_id,
                    predicted_labels=pred_row.predicted_labels or [],
                    uncertainty=pred_row.uncertainty,
                    diversity=pred_row.diversity,
                    density=pred_row.density,
                    composite_score=pred_row.composite_score,
                )
            )

            projections_2d["pca"].x.append(vis_row.pca_2d_x)
            projections_2d["pca"].y.append(vis_row.pca_2d_y)

            projections_2d["umap"].x.append(vis_row.umap_2d_x)
            projections_2d["umap"].y.append(vis_row.umap_2d_y)

            projections_2d["tsne"].x.append(vis_row.tsne_2d_x)
            projections_2d["tsne"].y.append(vis_row.tsne_2d_y)

            projections_2d["isomap"].x.append(vis_row.isomap_2d_x)
            projections_2d["isomap"].y.append(vis_row.isomap_2d_y)

            if (
                vis_row.pca_3d_x is not None and vis_row.pca_3d_y is not None and vis_row.pca_3d_z is not None
            ):
                has_any_3d = True
                projections_3d["pca"].x.append(vis_row.pca_3d_x)
                projections_3d["pca"].y.append(vis_row.pca_3d_y)
                projections_3d["pca"].z.append(vis_row.pca_3d_z)
            else:
                projections_3d["pca"].x.append(None)
                projections_3d["pca"].y.append(None)
                projections_3d["pca"].z.append(None)

            if (
                vis_row.umap_3d_x is not None and vis_row.umap_3d_y is not None and vis_row.umap_3d_z is not None
            ):
                has_any_3d = True
                projections_3d["umap"].x.append(vis_row.umap_3d_x)
                projections_3d["umap"].y.append(vis_row.umap_3d_y)
                projections_3d["umap"].z.append(vis_row.umap_3d_z)
            else:
                projections_3d["umap"].x.append(None)
                projections_3d["umap"].y.append(None)
                projections_3d["umap"].z.append(None)

            if (
                vis_row.tsne_3d_x is not None and vis_row.tsne_3d_y is not None and vis_row.tsne_3d_z is not None
            ):
                has_any_3d = True
                projections_3d["tsne"].x.append(vis_row.tsne_3d_x)
                projections_3d["tsne"].y.append(vis_row.tsne_3d_y)
                projections_3d["tsne"].z.append(vis_row.tsne_3d_z)
            else:
                projections_3d["tsne"].x.append(None)
                projections_3d["tsne"].y.append(None)
                projections_3d["tsne"].z.append(None)

            if (
                vis_row.isomap_3d_x is not None and vis_row.isomap_3d_y is not None and vis_row.isomap_3d_z is not None
            ):
                has_any_3d = True
                projections_3d["isomap"].x.append(vis_row.isomap_3d_x)
                projections_3d["isomap"].y.append(vis_row.isomap_3d_y)
                projections_3d["isomap"].z.append(vis_row.isomap_3d_z)
            else:
                projections_3d["isomap"].x.append(None)
                projections_3d["isomap"].y.append(None)
                projections_3d["isomap"].z.append(None)

        return FPVResponse(
            dataset_id=body.dataset_id,
            model_family_name=body.model_family_name,
            model_checkpoint_id=checkpoint_id,
            embedding_model_id=rows[0][0].embedding_model_id,
            color_filter_value=body.color_filter_value,
            visibility_filter_value=body.visibility_filter_value,
            color=FPVColorMetadata(
                field=body.color_filter_value,
                values=color_values,
                mode=color_mode,
            ),
            points=points,
            projections_2d=projections_2d,
            projections_3d=projections_3d if has_any_3d else None,
        )

    # ------------------------------------------------------------------
    # Dataset-level projections (computed from EmbeddingVector once)
    # ------------------------------------------------------------------
    def generate_fpv_for_dataset_embeddings(self, body: FPVDatasetRequest) -> FPVResponse:
        # DR is keyed by dataset_id and can span *all* SnippetSets under that
        # dataset for the given embedding_model_id, whereas AL's embedding
        # cache (app/services/pam_al/_embedding_cache.py) is keyed by a single
        # snippet_set_id. A dataset normally has exactly one SnippetSet per
        # embedding_model_id (SnippetSet models "dataset x embedding_model"),
        # but the schema doesn't enforce that -- re-segmentation can leave a
        # second row behind. Reusing the cache is only safe to do silently in
        # the common single-snippet-set case; if we find zero or more than
        # one, fall back to the direct query rather than risk silently
        # dropping snippets that belong to a second, non-default SnippetSet.
        snippet_set_ids = [
            row[0]
            for row in self.db.query(SnippetSet.id)
            .filter(SnippetSet.dataset_id == body.dataset_id)
            .filter(SnippetSet.embedding_model_id == body.embedding_model_id)
            .all()
        ]

        vectors: list
        snippet_ids: list[int]

        if len(snippet_set_ids) == 1:
            from app.services.pam_al._embedding_cache import load_embeddings_cached

            logger.info(
                "generate_fpv_for_dataset_embeddings: reusing AL embedding cache "
                "(snippet_set_id=%s, embedding_model_id=%s) for dataset_id=%s",
                snippet_set_ids[0], body.embedding_model_id, body.dataset_id,
            )
            X_cached, snippet_rows = load_embeddings_cached(
                self.db, snippet_set_ids[0], body.embedding_model_id,
            )
            X = np.asarray(X_cached, dtype=np.float32)
            snippet_ids = [row["snippet_id"] for row in snippet_rows]
        else:
            logger.info(
                "generate_fpv_for_dataset_embeddings: %d SnippetSet(s) found for "
                "dataset_id=%s, embedding_model_id=%s (expected exactly 1) -- "
                "skipping embedding cache, querying directly",
                len(snippet_set_ids), body.dataset_id, body.embedding_model_id,
            )
            # Column-only query (not full EmbeddingVector/Snippet ORM entities): at
            # dataset sizes in the hundreds of thousands to low millions of
            # snippets, hydrating two full ORM object graphs per row costs far
            # more time/memory than the vectors themselves. We only ever use
            # Snippet.id and EmbeddingVector.vector, so select just those.
            query = (
                self.db.query(EmbeddingVector.vector, Snippet.id)
                .join(Snippet, Snippet.id == EmbeddingVector.snippet_id)
                .join(SnippetSet, SnippetSet.id == Snippet.snippet_set_id)
                .filter(SnippetSet.dataset_id == body.dataset_id)
                .filter(EmbeddingVector.embedding_model_id == body.embedding_model_id)
                .order_by(Snippet.id.asc())
            )

            # Stream via a server-side cursor so the driver doesn't buffer the
            # entire result set client-side before we can start building X. Only
            # safe/supported on Postgres (tests run against sqlite in-memory).
            if self.db.get_bind().dialect.name == "postgresql":
                query = query.execution_options(stream_results=True)

            vectors = []
            snippet_ids = []
            for vector, snippet_id in query.yield_per(5000):
                vectors.append(vector)
                snippet_ids.append(snippet_id)

            if not vectors:
                raise ValueError(
                    f"No embeddings found for dataset_id={body.dataset_id}, "
                    f"embedding_model_id={body.embedding_model_id}."
                )

            X = np.array(vectors, dtype=np.float32)

        # Close the read transaction before long-running DR (PCA/UMAP/t-SNE).
        self.db.commit()

        coords = self._compute_visualizations(X=X, run_3d=body.run_3d)

        persist_fpv_vis_dataset_rows(
            dataset_id=body.dataset_id,
            embedding_model_id=body.embedding_model_id,
            snippet_ids=snippet_ids,
            coords=coords,
        )

        # Projections changed — drop any cached serialized responses so the next
        # request rebuilds (and re-warms) them.
        try:
            from app.services.fpv_cache import invalidate_fpv
            invalidate_fpv(body.dataset_id, body.embedding_model_id)
        except Exception:
            pass

        read_db = SessionLocal()
        try:
            return VISService(read_db).get_fpv_for_dataset_embeddings(body)
        finally:
            read_db.close()

    def get_fpv_for_dataset_embeddings(self, body: FPVDatasetRequest) -> FPVResponse:
        rows = (
            self.db.query(FPVVis)
            .filter(FPVVis.dataset_id == body.dataset_id)
            .filter(FPVVis.embedding_model_id == body.embedding_model_id)
            .filter(FPVVis.model_checkpoint_id.is_(None))
            .order_by(FPVVis.snippet_id.asc())
            .all()
        )

        if not rows:
            raise ValueError(
                f"No dataset-level feature projection rows found for dataset_id={body.dataset_id}, "
                f"embedding_model_id={body.embedding_model_id}. Generate projections first."
            )

        points = []
        selected_methods = self._selected_methods(body.method)
        projections_2d = {
            method: FPVProjection2D(x=[], y=[])
            for method in selected_methods
        }

        projections_3d = {
            method: FPVProjection3D(x=[], y=[], z=[])
            for method in selected_methods
        }

        has_any_3d = False

        for vis_row in rows:
            points.append(
                FPVPointMetadata(
                    snippet_id=vis_row.snippet_id,
                    predicted_labels=[],
                    uncertainty=None,
                    diversity=None,
                    density=None,
                    composite_score=None,
                )
            )

            for method in selected_methods:
                if method == "pca":
                    projections_2d[method].x.append(vis_row.pca_2d_x)
                    projections_2d[method].y.append(vis_row.pca_2d_y)
                    x3, y3, z3 = vis_row.pca_3d_x, vis_row.pca_3d_y, vis_row.pca_3d_z
                elif method == "umap":
                    projections_2d[method].x.append(vis_row.umap_2d_x)
                    projections_2d[method].y.append(vis_row.umap_2d_y)
                    x3, y3, z3 = vis_row.umap_3d_x, vis_row.umap_3d_y, vis_row.umap_3d_z
                elif method == "tsne":
                    projections_2d[method].x.append(vis_row.tsne_2d_x)
                    projections_2d[method].y.append(vis_row.tsne_2d_y)
                    x3, y3, z3 = vis_row.tsne_3d_x, vis_row.tsne_3d_y, vis_row.tsne_3d_z
                else:
                    projections_2d[method].x.append(vis_row.isomap_2d_x)
                    projections_2d[method].y.append(vis_row.isomap_2d_y)
                    x3, y3, z3 = vis_row.isomap_3d_x, vis_row.isomap_3d_y, vis_row.isomap_3d_z

                if x3 is not None and y3 is not None and z3 is not None:
                    has_any_3d = True
                    projections_3d[method].x.append(x3)
                    projections_3d[method].y.append(y3)
                    projections_3d[method].z.append(z3)
                else:
                    projections_3d[method].x.append(None)
                    projections_3d[method].y.append(None)
                    projections_3d[method].z.append(None)

        return FPVResponse(
            dataset_id=body.dataset_id,
            model_family_name=None,
            model_checkpoint_id=None,
            embedding_model_id=body.embedding_model_id,
            points=points,
            projections_2d=projections_2d,
            projections_3d=projections_3d if has_any_3d else None,
        )

    @staticmethod
    def _selected_methods(method: FPVMethod | None) -> list[str]:
        if method is None:
            return ["pca", "umap", "tsne", "isomap"]
        return [method.value]

    def _get_active_checkpoint(self, dataset_id: int, model_family_name: str) -> ALModelCheckpoint:
        family_state = (
            self.db.query(ALModelFamilyState)
            .filter(
                ALModelFamilyState.dataset_id == dataset_id,
                ALModelFamilyState.model_family_name == model_family_name,
            )
            .first()
        )

        if family_state is None or family_state.active_model_checkpoint_id is None:
            raise ValueError(
                f"No active checkpoint set for dataset_id={dataset_id}, "
                f"model_family_name='{model_family_name}'."
            )

        ckpt = (
            self.db.query(ALModelCheckpoint)
            .filter(ALModelCheckpoint.id == family_state.active_model_checkpoint_id)
            .first()
        )

        if ckpt is None:
            raise ValueError(f"Active checkpoint {family_state.active_model_checkpoint_id} not found.")

        return ckpt

    def _compute_visualizations(self, X: np.ndarray, run_3d: bool) -> dict:
        coords = {}
        n = int(X.shape[0])

        if n < 2:
            raise ValueError("Need at least 2 samples to compute visualization coordinates.")

        # t-SNE / Isomap are very slow for large n and often OOM on limited workers.
        # UMAP uses PCA pre-reduction (1024→50 dims) so it's fine at any n.
        full_dr_max_points = 15_000
        run_tsne_isomap = n <= full_dr_max_points

        logger.info(
            "fpv dataset: DR start n=%s run_3d=%s tsne_isomap=%s",
            n,
            run_3d,
            run_tsne_isomap,
        )

        if not run_tsne_isomap:
            logger.warning(
                "fpv dataset: skipping t-SNE and Isomap for n=%s (limit %s)",
                n,
                full_dr_max_points,
            )

        # Single PCA: covers pca_2d/3d slices AND pre-reduces for kNN methods.
        max_vis_dims = 3 if (run_3d and n >= 3 and X.shape[1] >= 3) else 2
        X_r = pre_reduce_pca(X, max_vis_dims=max_vis_dims)

        coords["pca_2d"] = X_r[:, :2]
        logger.info("fpv dataset: PCA 2D done n=%s", n)

        # Build shared kNN graph once on the already-reduced space.
        # k=90 satisfies all three methods:
        #   UMAP      needs k ≥ n_neighbors (90)
        #   t-SNE     needs k ≥ 3 × perplexity (3 × 30 = 90)
        #   Isomap    needs k ≥ n_neighbors (90), dense enough to avoid disconnected geodesic graph
        # This is cheaper than the original code which built three separate graphs
        # totalling ~151 neighbours (91 for t-SNE + 30 for UMAP + 30 for Isomap).
        _knn_neighbors = min(90, n - 1)
        knn = build_knn_graph(X_r, n_neighbors=_knn_neighbors)
        logger.info("fpv dataset: kNN graph built n=%s k=%s", n, _knn_neighbors)

        # PCA coords are free slices; the remaining methods run in parallel.
        if run_3d and n >= 3 and X.shape[1] >= 3:
            coords["pca_3d"] = X_r[:, :3]

        dr_tasks: dict[str, object] = {
            "umap_2d": lambda: run_dr_umap(X_r, dimensions=2, n_neighbors=_knn_neighbors, low_memory=True, precomputed_knn=knn),
        }
        if run_tsne_isomap:
            dr_tasks["tsne_2d"] = lambda: run_dr_tsne(X_r, dimensions=2, precomputed_knn=knn)
            dr_tasks["isomap_2d"] = lambda: run_dr_isomap(X_r, dimensions=2, n_neighbors=_knn_neighbors, precomputed_knn=knn)
        if run_3d and n >= 3:
            dr_tasks["umap_3d"] = lambda: run_dr_umap(X_r, dimensions=3, n_neighbors=_knn_neighbors, low_memory=True, precomputed_knn=knn)
        if run_3d and run_tsne_isomap and n >= 4:
            dr_tasks["tsne_3d"] = lambda: run_dr_tsne(X_r, dimensions=3, precomputed_knn=knn)
            dr_tasks["isomap_3d"] = lambda: run_dr_isomap(X_r, dimensions=3, n_neighbors=_knn_neighbors, precomputed_knn=knn)

        with ThreadPoolExecutor(max_workers=len(dr_tasks)) as executor:
            futures = {executor.submit(fn): name for name, fn in dr_tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    coords[name] = future.result()
                    logger.info("fpv dataset: %s done n=%s", name, n)
                except Exception:
                    logger.exception("fpv dataset: %s failed n=%s, skipping", name, n)

        logger.info("fpv dataset: DR finished n=%s methods=%s", n, list(coords.keys()))
        return coords

    def _upsert_fpv_vis_rows(
        self,
        dataset_id: int,
        model_checkpoint_id: int,
        embedding_model_id: int,
        snippet_ids: list[int],
        coords: dict,
    ) -> None:
        existing_rows = (
            self.db.query(FPVVis)
            .filter(FPVVis.model_checkpoint_id == model_checkpoint_id)
            .all()
        )
        existing_by_snippet = {row.snippet_id: row for row in existing_rows}

        for i, snippet_id in enumerate(snippet_ids):
            row = existing_by_snippet.get(snippet_id)
            if row is None:
                row.embedding_model_id = embedding_model_id
                row = FPVVis(
                    dataset_id=dataset_id,
                    model_checkpoint_id=model_checkpoint_id,
                    embedding_model_id=embedding_model_id,
                    snippet_id=snippet_id,
                )
                self.db.add(row)

            if "pca_2d" in coords:
                row.pca_2d_x = float(coords["pca_2d"][i, 0])
                row.pca_2d_y = float(coords["pca_2d"][i, 1])

            if "pca_3d" in coords:
                row.pca_3d_x = float(coords["pca_3d"][i, 0])
                row.pca_3d_y = float(coords["pca_3d"][i, 1])
                row.pca_3d_z = float(coords["pca_3d"][i, 2])

            if "umap_2d" in coords:
                row.umap_2d_x = float(coords["umap_2d"][i, 0])
                row.umap_2d_y = float(coords["umap_2d"][i, 1])

            if "umap_3d" in coords:
                row.umap_3d_x = float(coords["umap_3d"][i, 0])
                row.umap_3d_y = float(coords["umap_3d"][i, 1])
                row.umap_3d_z = float(coords["umap_3d"][i, 2])

            if "tsne_2d" in coords:
                row.tsne_2d_x = float(coords["tsne_2d"][i, 0])
                row.tsne_2d_y = float(coords["tsne_2d"][i, 1])

            if "tsne_3d" in coords:
                row.tsne_3d_x = float(coords["tsne_3d"][i, 0])
                row.tsne_3d_y = float(coords["tsne_3d"][i, 1])
                row.tsne_3d_z = float(coords["tsne_3d"][i, 2])

            if "isomap_2d" in coords:
                row.isomap_2d_x = float(coords["isomap_2d"][i, 0])
                row.isomap_2d_y = float(coords["isomap_2d"][i, 1])

            if "isomap_3d" in coords:
                row.isomap_3d_x = float(coords["isomap_3d"][i, 0])
                row.isomap_3d_y = float(coords["isomap_3d"][i, 1])
                row.isomap_3d_z = float(coords["isomap_3d"][i, 2])

    def _upsert_fpv_vis_dataset_rows(
        self,
        dataset_id: int,
        embedding_model_id: int,
        snippet_ids: list[int],
        coords: dict,
    ) -> None:
        existing_rows = (
            self.db.query(FPVVis)
            .filter(FPVVis.dataset_id == dataset_id)
            .filter(FPVVis.embedding_model_id == embedding_model_id)
            .filter(FPVVis.model_checkpoint_id.is_(None))
            .all()
        )
        existing_by_snippet = {row.snippet_id: row for row in existing_rows}

        for i, snippet_id in enumerate(snippet_ids):
            row = existing_by_snippet.get(snippet_id)
            if row is None:
                row = FPVVis(
                    dataset_id=dataset_id,
                    model_checkpoint_id=None,
                    embedding_model_id=embedding_model_id,
                    snippet_id=snippet_id,
                )
                self.db.add(row)

            if "pca_2d" in coords:
                row.pca_2d_x = float(coords["pca_2d"][i, 0])
                row.pca_2d_y = float(coords["pca_2d"][i, 1])

            if "pca_3d" in coords:
                row.pca_3d_x = float(coords["pca_3d"][i, 0])
                row.pca_3d_y = float(coords["pca_3d"][i, 1])
                row.pca_3d_z = float(coords["pca_3d"][i, 2])

            if "umap_2d" in coords:
                row.umap_2d_x = float(coords["umap_2d"][i, 0])
                row.umap_2d_y = float(coords["umap_2d"][i, 1])

            if "umap_3d" in coords:
                row.umap_3d_x = float(coords["umap_3d"][i, 0])
                row.umap_3d_y = float(coords["umap_3d"][i, 1])
                row.umap_3d_z = float(coords["umap_3d"][i, 2])

            if "tsne_2d" in coords:
                row.tsne_2d_x = float(coords["tsne_2d"][i, 0])
                row.tsne_2d_y = float(coords["tsne_2d"][i, 1])

            if "tsne_3d" in coords:
                row.tsne_3d_x = float(coords["tsne_3d"][i, 0])
                row.tsne_3d_y = float(coords["tsne_3d"][i, 1])
                row.tsne_3d_z = float(coords["tsne_3d"][i, 2])

            if "isomap_2d" in coords:
                row.isomap_2d_x = float(coords["isomap_2d"][i, 0])
                row.isomap_2d_y = float(coords["isomap_2d"][i, 1])

            if "isomap_3d" in coords:
                row.isomap_3d_x = float(coords["isomap_3d"][i, 0])
                row.isomap_3d_y = float(coords["isomap_3d"][i, 1])
                row.isomap_3d_z = float(coords["isomap_3d"][i, 2])

    def get_fpv_vis_range(self, visibility_filter_value: FPVVisibilityField) -> FPVVisibilityRangeResponse:
        if visibility_filter_value in {
            FPVVisibilityField.UNCERTAINTY,
            FPVVisibilityField.DIVERSITY,
            FPVVisibilityField.DENSITY,
            FPVVisibilityField.COMPOSITE,
        }:
            return FPVVisibilityRangeResponse(
                field=visibility_filter_value,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                label="score",
            )

        if visibility_filter_value == FPVVisibilityField.YEAR_CYCLE:
            return FPVVisibilityRangeResponse(
                field=visibility_filter_value,
                min_value=1,
                max_value=12,
                step=1,
                label="month",
            )

        if visibility_filter_value == FPVVisibilityField.DAY_CYCLE:
            return FPVVisibilityRangeResponse(
                field=visibility_filter_value,
                min_value=0,
                max_value=23,
                step=1,
                label="hour (0–23)",
            )

        return FPVVisibilityRangeResponse(
            field=visibility_filter_value,
            min_value=0,
            max_value=0,
            step=1,
            label="none",
        )

    def _passes_visibility_filter(self, pred_row: ALPrediction, body: FPVRequest) -> bool:
        field = body.visibility_filter_value

        if field == FPVVisibilityField.NONE:
            return True

        min_v = body.visibility_range_min
        max_v = body.visibility_range_max

        if min_v is None or max_v is None:
            return True

        value = None

        if field == FPVVisibilityField.UNCERTAINTY:
            value = pred_row.uncertainty
        elif field == FPVVisibilityField.DIVERSITY:
            value = pred_row.diversity
        elif field == FPVVisibilityField.DENSITY:
            value = pred_row.density
        elif field == FPVVisibilityField.COMPOSITE:
            value = pred_row.composite_score
        elif field == FPVVisibilityField.YEAR_CYCLE:
            # metadata not implemented yet
            return True
        elif field == FPVVisibilityField.DAY_CYCLE:
            # metadata not implemented yet
            return True

        if value is None:
            return False

        return min_v <= value <= max_v

    def _build_color_values(self, pred_row: ALPrediction, color_field: FPVColorField):
        if color_field == FPVColorField.NONE:
            return None, "none"

        if color_field == FPVColorField.PREDICTED_LABEL:
            labels = pred_row.predicted_labels or []
            return (labels[0] if labels else None), "categorical"

        if color_field == FPVColorField.UNCERTAINTY:
            return pred_row.uncertainty, "continuous"

        if color_field == FPVColorField.DIVERSITY:
            return pred_row.diversity, "continuous"

        if color_field == FPVColorField.DENSITY:
            return pred_row.density, "continuous"

        if color_field == FPVColorField.COMPOSITE:
            return pred_row.composite_score, "continuous"

        # Not implemented yet because metadata is not available
        if color_field in {
            FPVColorField.YEAR_CYCLE,
            FPVColorField.DAY_CYCLE,
            FPVColorField.SOUND_TYPE,
            FPVColorField.BIRDNET_LABEL,
            FPVColorField.YAMNET_LABEL,
        }:
            return None, "categorical"

        return None, "none"

    # ONLY FOR LINEAR CLASSIFIERS HAVING NO INTERMEDIATE EMBEDDINGS
    def _copy_dataset_fpv_to_checkpoint(
            self,
            dataset_id: int,
            model_ckpt: ALModelCheckpoint,
            run_3d: bool,
    ) -> None:
        hyper = model_ckpt.hyperparameters or {}
        embedding_model_id = hyper.get("embedding_model_id")

        if embedding_model_id is None:
            raise ValueError(f"Checkpoint {model_ckpt.id} missing embedding_model_id.")

        dataset_body = FPVDatasetRequest(
            dataset_id=dataset_id,
            embedding_model_id=embedding_model_id,
            run_3d=run_3d,
        )

        try:
            self.get_fpv_for_dataset_embeddings(dataset_body)
        except ValueError:
            self.generate_fpv_for_dataset_embeddings(dataset_body)

        source_rows = (
            self.db.query(FPVVis)
            .filter(FPVVis.dataset_id == dataset_id)
            .filter(FPVVis.embedding_model_id == embedding_model_id)
            .filter(FPVVis.model_checkpoint_id.is_(None))
            .order_by(FPVVis.snippet_id.asc())
            .all()
        )

        if not source_rows:
            raise ValueError(
                f"No dataset-level FPV rows available after generation for "
                f"dataset_id={dataset_id}, embedding_model_id={embedding_model_id}."
            )

        existing_rows = (
            self.db.query(FPVVis)
            .filter(FPVVis.dataset_id == dataset_id)
            .filter(FPVVis.model_checkpoint_id == model_ckpt.id)
            .all()
        )
        existing_by_snippet = {row.snippet_id: row for row in existing_rows}

        for src in source_rows:
            row = existing_by_snippet.get(src.snippet_id)

            if row is None:
                row = FPVVis(
                    dataset_id=dataset_id,
                    model_checkpoint_id=model_ckpt.id,
                    embedding_model_id=embedding_model_id,
                    snippet_id=src.snippet_id,
                )
                self.db.add(row)

            row.embedding_model_id = embedding_model_id

            row.pca_2d_x = src.pca_2d_x
            row.pca_2d_y = src.pca_2d_y
            row.umap_2d_x = src.umap_2d_x
            row.umap_2d_y = src.umap_2d_y
            row.tsne_2d_x = src.tsne_2d_x
            row.tsne_2d_y = src.tsne_2d_y
            row.isomap_2d_x = src.isomap_2d_x
            row.isomap_2d_y = src.isomap_2d_y

            if run_3d:
                row.pca_3d_x = src.pca_3d_x
                row.pca_3d_y = src.pca_3d_y
                row.pca_3d_z = src.pca_3d_z

                row.umap_3d_x = src.umap_3d_x
                row.umap_3d_y = src.umap_3d_y
                row.umap_3d_z = src.umap_3d_z

                row.tsne_3d_x = src.tsne_3d_x
                row.tsne_3d_y = src.tsne_3d_y
                row.tsne_3d_z = src.tsne_3d_z

                row.isomap_3d_x = src.isomap_3d_x
                row.isomap_3d_y = src.isomap_3d_y
                row.isomap_3d_z = src.isomap_3d_z

        self.db.commit()
