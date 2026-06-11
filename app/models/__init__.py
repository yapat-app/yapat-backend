"""
SQLAlchemy models
"""

from app.models.user import User
from app.models.team import Team, TeamMembership
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.annotation import Annotation
from app.models.invitation import InvitationLink
from app.models.embedding import (
    EmbeddingModel,
    EmbeddingJob,
    EmbeddingVector,
    SnippetSet,
)
from app.models.user_feed import UserFeed
from app.models.custom_taxonomy import CustomTaxonomy, TaxonomyStatus
from app.models.taxonomy_conversation import (
    TaxonomyConversation,
    TaxonomyMessage,
    ConversationStatus,
    MessageRole,
)
from app.models.wssed import (
    WSSEDTrainingJob,
    WSSEDPrediction,
    WSSEDStrongLabel,
    WSSEDSpeciesModel,
    WSSEDSnippetLabel,
    TrainingStatus,
    FeedbackType,
)
from app.models.pam_active_learning import (
    ALModelCheckpoint,
    ALPrediction,
    ALFeedbackEvent,
    ALRetrainJob,
    ALModelStatus,
    ALFeedbackAction,
    ALRetrainStatus,
)
from app.models.study_event import StudyEvent

__all__ = [
    "User",
    "Team",
    "TeamMembership",
    "Dataset",
    "Recording",
    "Snippet",
    "SnippetSet",
    "EmbeddingModel",
    "EmbeddingJob",
    "Annotation",
    "InvitationLink",
    "UserFeed",
    "CustomTaxonomy",
    "TaxonomyStatus",
    "TaxonomyConversation",
    "TaxonomyMessage",
    "ConversationStatus",
    "MessageRole",
    "WSSEDTrainingJob",
    "WSSEDPrediction",
    "WSSEDStrongLabel",
    "WSSEDSpeciesModel",
    "WSSEDSnippetLabel",
    "TrainingStatus",
    "FeedbackType",
    # PAM Active Learning
    "ALModelCheckpoint",
    "ALPrediction",
    "ALFeedbackEvent",
    "ALRetrainJob",
    "ALModelStatus",
    "ALFeedbackAction",
    "ALRetrainStatus",
    # Study interaction logging
    "StudyEvent",
]
