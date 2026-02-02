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
    team_id: int = Field(..., description="Team ID for the conversation")


class LabelSpaceItem(BaseModel):
    """Single item in the label space list"""
    id: str = Field(..., description="Unique ID for this item")
    name: str = Field(..., description="Display name (e.g., 'Screaming Piha')")
    scientific_name: Optional[str] = Field(None, description="Scientific name (e.g., 'Lipaugus vociferans')")
    taxon_id: Optional[str] = Field(None, description="GBIF taxon ID if available (e.g., 'gbif:2482715')")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (family, rank, etc.)")
    added_at: datetime = Field(..., description="When this item was added to the list")


class ConversationResponse(BaseModel):
    """Response schema for a conversation (label space building session)"""
    id: int
    team_id: int
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
    """Response for chat message with generated taxonomy"""
    message: MessageResponse = Field(..., description="Assistant's response message")
    generated_taxonomy: Optional[Dict[str, Any]] = Field(None, description="Generated taxonomy data if available")
    conversation: ConversationResponse = Field(..., description="Updated conversation state")


class AddToLabelSpaceRequest(BaseModel):
    """Request for adding a species to the label space"""
    # No fields needed - the species info comes from the last assistant message


class AddToLabelSpaceResponse(BaseModel):
    """Response after adding to label space"""
    conversation: ConversationResponse = Field(..., description="Updated conversation with new item in label_space")
    added_item: LabelSpaceItem = Field(..., description="The item that was added")


class FreezeLabelSpaceRequest(BaseModel):
    """Request for freezing the label space and creating taxonomy"""
    name: str = Field(..., min_length=1, max_length=255, description="Name for the custom taxonomy")
    description: Optional[str] = Field(None, description="Description of the taxonomy")


class FreezeLabelSpaceResponse(BaseModel):
    """Response after freezing label space"""
    conversation: ConversationResponse = Field(..., description="Frozen conversation")
    taxonomy: "CustomTaxonomyResponse" = Field(..., description="Created custom taxonomy")


class ConversationListResponse(BaseModel):
    """List response for conversations"""
    conversations: List[ConversationResponse]
    total: int


# Resolve forward references after all imports are done
def _rebuild_models():
    """Rebuild models to resolve forward references"""
    from app.schemas.custom_taxonomy import CustomTaxonomyResponse
    FreezeLabelSpaceResponse.model_rebuild()

_rebuild_models()
