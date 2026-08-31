"""
Taxonomy Conversation schemas
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List


class MessageCreate(BaseModel):
    """Schema for creating a message"""
    content: str = Field(..., min_length=1, max_length=5000, description="Message content")


class MessageResponse(BaseModel):
    """Response schema for a message"""
    id: int
    conversation_id: int
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str
    metadata: Optional[Dict[str, Any]] = Field(None, alias="message_metadata", description="Message metadata (taxonomy_data, etc.)")

    class Config:
        from_attributes = True
        populate_by_name = True


class ConversationCreate(BaseModel):
    """Schema for creating a conversation"""
    team_id: Optional[int] = Field(None, description="Team ID for the conversation. If not provided, the user's first team is used (or derived from dataset_id).")
    dataset_id: Optional[int] = Field(None, description="Dataset ID to associate the label space with. Labels added from chat are mirrored into this dataset's quick_labels. When provided, team_id is derived from the dataset's team if not given explicitly.")
    seed_from_active: bool = Field(False, description="When true, pre-populate the new label space with the team's current active label-space version so the user edits a copy of it.")


class LabelSpaceItem(BaseModel):
    """Single item in the label space list"""
    id: str = Field(..., description="Unique ID for this item")
    name: str = Field(..., description="Display name (e.g., 'Screaming Piha')")
    scientific_name: Optional[str] = Field(None, description="Scientific name (e.g., 'Lipaugus vociferans')")
    taxon_id: Optional[str] = Field(None, description="GBIF taxon ID if available (e.g., 'gbif:2482715')")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (family, rank, etc.)")
    added_at: datetime = Field(..., description="When this item was added to the list")
    added_by_user_id: Optional[int] = Field(None, description="ID of the user who added this label")
    added_by_username: Optional[str] = Field(None, description="Username snapshot of who added this label (may be null for legacy items)")


class ConversationResponse(BaseModel):
    """Response schema for a conversation (label space building session)"""
    id: int
    team_id: Optional[int] = None
    dataset_id: Optional[int] = None
    user_id: int
    custom_taxonomy_id: Optional[int] = None
    status: str = Field(..., description="Conversation status: in_progress, completed, cancelled")
    label_space: List[LabelSpaceItem] = Field(default_factory=list, description="Accumulated list of species/taxa")
    is_frozen: bool = Field(False, description="Whether the label space is frozen")
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: List[MessageResponse] = Field(default_factory=list, description="Conversation messages")

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """Request for sending a chat message"""
    prompt: str = Field(..., min_length=10, max_length=2000, description="User prompt for taxonomy generation")


class ChatResponse(BaseModel):
    """Response for chat message. Taxonomy for this turn is in message.message_metadata.taxonomy_data (nodes + metadata)."""
    message: MessageResponse = Field(..., description="The new assistant message; use message_metadata.taxonomy_data.nodes for species list (indices 1, 2, 3...)")
    conversation: ConversationResponse = Field(..., description="Updated conversation state (label_space, messages). Latest message in messages omits full taxonomy to avoid duplication.")


class AddToLabelSpaceRequest(BaseModel):
    """Request for adding species to the label space"""
    message_id: Optional[int] = Field(None, description="ID of the assistant message to add from. If not provided, uses the last assistant message.")
    indices: Optional[List[int]] = Field(None, description="1-based indices of specific species to add (e.g., [1, 2, 3]). If not provided, adds all species from the message.")


class AddToLabelSpaceResponse(BaseModel):
    """Response after adding to label space"""
    conversation: ConversationResponse = Field(..., description="Updated conversation with new item(s) in label_space")
    added_items: List[LabelSpaceItem] = Field(..., description="The item(s) that were added")
    skipped_count: int = Field(0, description="Number of items that were skipped (duplicates or invalid indices)")


class FreezeLabelSpaceRequest(BaseModel):
    """Request for freezing the label space and creating taxonomy"""
    name: str = Field(..., min_length=1, max_length=255, description="Name for the custom taxonomy")
    description: Optional[str] = Field(None, description="Description of the taxonomy")


class FreezeLabelSpaceResponse(BaseModel):
    """Response after freezing label space"""
    conversation: ConversationResponse = Field(..., description="Frozen conversation")
    taxonomy: "CustomTaxonomyResponse" = Field(..., description="Created custom taxonomy")


class SubmitLabelSpaceRequest(BaseModel):
    """Request for finalizing (submitting) a label space version to the team owner"""
    description: Optional[str] = Field(None, description="Optional description of this label-space version")


class SubmitLabelSpaceResponse(BaseModel):
    """Response after finalizing/submitting a label space version"""
    conversation: ConversationResponse = Field(..., description="Finalized conversation")
    taxonomy: "CustomTaxonomyResponse" = Field(..., description="Created (submitted) label-space version")


class ConversationListResponse(BaseModel):
    """List response for conversations"""
    conversations: List[ConversationResponse]
    total: int


class LabelSpaceResponse(BaseModel):
    """Response for label space items"""
    conversation_id: int = Field(..., description="ID of the conversation")
    is_frozen: bool = Field(..., description="Whether the label space is frozen")
    items: List[LabelSpaceItem] = Field(default_factory=list, description="List of items in the label space")
    total: int = Field(..., description="Total number of items in the label space")


# Resolve forward references after all imports are done
def _rebuild_models():
    """Rebuild models to resolve forward references"""
    from app.schemas.custom_taxonomy import CustomTaxonomyResponse
    FreezeLabelSpaceResponse.model_rebuild()
    SubmitLabelSpaceResponse.model_rebuild()

_rebuild_models()
