"""
Active Learning package

Exports the PAM-specific active learning flow.
"""

from active_learning.pam_scoring import combined_score, select_top_k
from active_learning.pam_classifier import (
    PAMClassifierProtocol,
    PAMClassifierStub,
    load_pam_classifier,
)
from active_learning.pam_model_checkout import PAMModelHandle, checkout_model
from active_learning.pam_retrain import (
    InteractionCounter,
    get_interaction_counter,
    run_retrain,
    save_retrained_checkpoint,
    AUTO_RETRAIN_THRESHOLD,
)

__all__ = [
    # Scoring
    "combined_score",
    "select_top_k",
    # Classifier
    "PAMClassifierProtocol",
    "PAMClassifierStub",
    "load_pam_classifier",
    # Model checkout
    "PAMModelHandle",
    "checkout_model",
    # Retrain
    "InteractionCounter",
    "get_interaction_counter",
    "run_retrain",
    "save_retrained_checkpoint",
    "AUTO_RETRAIN_THRESHOLD",
]
