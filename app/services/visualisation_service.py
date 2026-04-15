import numpy as np
from sqlalchemy.orm import Session

from app.models.pam_active_learning import ALPrediction, ALModelFamilyState
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
)
from utils.dr_methods import run_dr_isomap, run_dr_pca, run_dr_tsne, run_dr_umap


class VISService:
    def __init__(self, db: Session):
        self.db = db

    def generate_fpv_for_checkpoint(self, body: FPVRequest) -> FPVResponse:
        checkpoint_id = self._get_active_checkpoint_id(
            dataset_id=body.dataset_id,
            model_family_name=body.model_family_name,
        )

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

        coords = self._compute_visualizations(X=X, run_3d=body.run_3d)

        self._upsert_fpv_vis_rows(
            dataset_id=body.dataset_id,
            model_checkpoint_id=checkpoint_id,
            snippet_ids=snippet_ids,
            coords=coords,
        )

        self.db.commit()

        return self.get_fpv_for_checkpoint(body)

    def get_fpv_for_checkpoint(self, body: FPVRequest) -> FPVResponse:
        checkpoint_id = self._get_active_checkpoint_id(
            dataset_id=body.dataset_id,
            model_family_name=body.model_family_name,
        )

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

        for vis_row, pred_row in rows:
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
            embedding_model_id=None,
            points=points,
            projections_2d=projections_2d,
            projections_3d=projections_3d if has_any_3d else None,
        )

    # ------------------------------------------------------------------
    # Dataset-level projections (computed from EmbeddingVector once)
    # ------------------------------------------------------------------
    def generate_fpv_for_dataset_embeddings(self, body: FPVDatasetRequest) -> FPVResponse:
        rows = (
            self.db.query(EmbeddingVector, Snippet)
            .join(Snippet, Snippet.id == EmbeddingVector.snippet_id)
            .join(SnippetSet, SnippetSet.id == Snippet.snippet_set_id)
            .filter(SnippetSet.dataset_id == body.dataset_id)
            .filter(EmbeddingVector.embedding_model_id == body.embedding_model_id)
            .order_by(Snippet.id.asc())
            .all()
        )

        if not rows:
            raise ValueError(
                f"No embeddings found for dataset_id={body.dataset_id}, "
                f"embedding_model_id={body.embedding_model_id}."
            )

        snippet_ids = [s.id for (_, s) in rows]
        X = np.array([ev.vector for (ev, _) in rows], dtype=np.float32)

        coords = self._compute_visualizations(X=X, run_3d=body.run_3d)

        self._upsert_fpv_vis_dataset_rows(
            dataset_id=body.dataset_id,
            embedding_model_id=body.embedding_model_id,
            snippet_ids=snippet_ids,
            coords=coords,
        )

        self.db.commit()
        return self.get_fpv_for_dataset_embeddings(body)

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
            model_family_name=None,
            model_checkpoint_id=None,
            embedding_model_id=body.embedding_model_id,
            points=points,
            projections_2d=projections_2d,
            projections_3d=projections_3d if has_any_3d else None,
        )

    def _get_active_checkpoint_id(self, dataset_id: int, model_family_name: str) -> int:
        family_state = (
            self.db.query(ALModelFamilyState)
            .filter(
                ALModelFamilyState.dataset_id == dataset_id,
                ALModelFamilyState.model_family_name == model_family_name,
            )
            .first()
        )

        if family_state is None:
            raise ValueError(
                f"No model family state found for dataset_id={dataset_id}, "
                f"model_family_name='{model_family_name}'."
            )

        checkpoint_id = family_state.active_model_checkpoint_id
        if checkpoint_id is None:
            raise ValueError(
                f"No active checkpoint set for dataset_id={dataset_id}, "
                f"model_family_name='{model_family_name}'."
            )

        return checkpoint_id

    def _compute_visualizations(self, X: np.ndarray, run_3d: bool) -> dict:
        coords = {}

        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples to compute visualization coordinates.")

        coords["pca_2d"] = run_dr_pca(X, dimensions=2)
        coords["umap_2d"] = run_dr_umap(X, dimensions=2)

        #perplexity = max(2, min(30, X.shape[0] - 1))
        coords["tsne_2d"] = run_dr_tsne(X, dimensions=2)

        #n_neighbors = max(2, min(5, X.shape[0] - 1))
        coords["isomap_2d"] = run_dr_isomap(X, dimensions=2)

        if run_3d and X.shape[0] >= 3 and X.shape[1] >= 3:
            coords["pca_3d"] = run_dr_pca(X, dimensions=3)

        if run_3d and X.shape[0] >= 3:
            coords["umap_3d"] = run_dr_umap(X, dimensions=3)

        if run_3d and X.shape[0] >= 4:
            coords["tsne_3d"] = run_dr_tsne(X, dimensions=3)
            coords["isomap_3d"] = run_dr_isomap(X, dimensions=3)

        return coords

    def _upsert_fpv_vis_rows(
        self,
        dataset_id: int,
        model_checkpoint_id: int,
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
                row = FPVVis(
                    dataset_id=dataset_id,
                    model_checkpoint_id=model_checkpoint_id,
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