"""
Freeze label space functionality for custom taxonomy service
"""

import logging
import uuid
from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.models import (
    CustomTaxonomy,
    TaxonomyConversation,
    TaxonomyMessage,
    ConversationStatus,
    MessageRole,
    TaxonomyStatus,
)


logger = logging.getLogger(__name__)


class CustomTaxonomyServiceError(Exception):
    """Base exception for custom taxonomy service errors"""
    pass


def freeze_label_space(
    conversation_id: int,
    user_id: int,
    name: str,
    db: Session,
    description: Optional[str] = None
) -> dict:
    """
    Freeze the label space and create a CustomTaxonomy from the accumulated list.
    
    Args:
        conversation_id: ID of the conversation
        user_id: ID of the user freezing the label space
        name: Name for the custom taxonomy
        db: Database session
        description: Optional description for the taxonomy
        
    Returns:
        Dict with conversation and created taxonomy
        
    Raises:
        CustomTaxonomyServiceError: If conversation not found, empty, or already frozen
    """
    # Get conversation
    conversation = db.query(TaxonomyConversation).filter(
        TaxonomyConversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise CustomTaxonomyServiceError(f"Conversation {conversation_id} not found")
    
    if conversation.is_frozen:
        raise CustomTaxonomyServiceError("Label space is already frozen")
    
    if conversation.status != ConversationStatus.IN_PROGRESS:
        raise CustomTaxonomyServiceError("Conversation is not in progress")
    
    # Check if label_space has items
    if not conversation.label_space or len(conversation.label_space) == 0:
        raise CustomTaxonomyServiceError("Label space is empty. Add at least one species before freezing.")
    
    # Generate unique taxonomy ID
    taxonomy_id = f"custom:{uuid.uuid4()}"
    
    # Build taxonomy_data from label_space
    # Structure it as a flat list of nodes for simplicity
    taxonomy_data = {
        "nodes": conversation.label_space,
        "metadata": {
            "created_from_conversation": conversation_id,
            "total_species": len(conversation.label_space),
            "created_at": datetime.utcnow().isoformat()
        }
    }
    
    # Create CustomTaxonomy
    custom_taxonomy = CustomTaxonomy(
        taxonomy_id=taxonomy_id,
        team_id=conversation.team_id,
        created_by_user_id=user_id,
        name=name,
        description=description,
        taxonomy_data=taxonomy_data,
        status=TaxonomyStatus.ACTIVE,
        is_global=False
    )
    
    db.add(custom_taxonomy)
    
    # Update conversation - mark as frozen and completed
    conversation.is_frozen = True
    conversation.status = ConversationStatus.COMPLETED
    conversation.custom_taxonomy_id = custom_taxonomy.id
    conversation.updated_at = datetime.utcnow()
    
    # Add completion message
    from app.services.custom_taxonomy_service import add_message
    
    add_message(
        conversation_id=conversation_id,
        role=MessageRole.SYSTEM,
        content=f"🎯 Label space frozen! Taxonomy '{name}' with {len(conversation.label_space)} species has been created and is now available for annotation.",
        db=db,
        metadata={"action": "frozen", "taxonomy_id": taxonomy_id}
    )
    
    db.commit()
    db.refresh(custom_taxonomy)
    db.refresh(conversation)
    
    logger.info(f"Froze label space and created taxonomy {taxonomy_id} from conversation {conversation_id}")
    
    return {
        "conversation": conversation,
        "taxonomy": custom_taxonomy
    }
