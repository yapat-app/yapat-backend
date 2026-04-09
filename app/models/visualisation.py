from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Float, Text, JSON,
    Enum as SQLEnum, UniqueConstraint,
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import func
import enum
from app.database import Base

# ── Fields required for Feature Projection ─────────────────────────────────────────────────────


class FPVVis(Base):
    __tablename__ = "fpv_vis"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_checkpoint_id = Column(
        Integer,
        ForeignKey("al_model_checkpoints.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    snippet_id = Column(
        Integer,
        ForeignKey("snippets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # PCA
    pca_2d_x = Column(Float, nullable=True)
    pca_2d_y = Column(Float, nullable=True)
    pca_3d_x = Column(Float, nullable=True)
    pca_3d_y = Column(Float, nullable=True)
    pca_3d_z = Column(Float, nullable=True)

    # UMAP
    umap_2d_x = Column(Float, nullable=True)
    umap_2d_y = Column(Float, nullable=True)
    umap_3d_x = Column(Float, nullable=True)
    umap_3d_y = Column(Float, nullable=True)
    umap_3d_z = Column(Float, nullable=True)

    # t-SNE
    tsne_2d_x = Column(Float, nullable=True)
    tsne_2d_y = Column(Float, nullable=True)
    tsne_3d_x = Column(Float, nullable=True)
    tsne_3d_y = Column(Float, nullable=True)
    tsne_3d_z = Column(Float, nullable=True)

    # Isomap
    isomap_2d_x = Column(Float, nullable=True)
    isomap_2d_y = Column(Float, nullable=True)
    isomap_3d_x = Column(Float, nullable=True)
    isomap_3d_y = Column(Float, nullable=True)
    isomap_3d_z = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    model_checkpoint = relationship("ALModelCheckpoint")
    snippet = relationship("Snippet")

    __table_args__ = (
        UniqueConstraint(
            "model_checkpoint_id",
            "snippet_id",
            name="uq_fpv_vis_checkpoint_snippet",
        ),
    )