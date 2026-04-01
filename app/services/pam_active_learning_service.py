"""Backward-compatible re-export.  All logic now lives in app.services.pam_al."""

from app.services.pam_al.service import PAMActiveLearningService  # noqa: F401

__all__ = ["PAMActiveLearningService"]
