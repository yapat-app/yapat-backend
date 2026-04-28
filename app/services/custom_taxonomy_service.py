"""
Custom Taxonomy Service

Business logic for custom taxonomy and conversation management.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import or_, and_
from datetime import datetime

from app.models import (
    CustomTaxonomy,
    TaxonomyConversation,
    TaxonomyMessage,
    ConversationStatus,
    MessageRole,
    TaxonomyStatus,
    Team,
    TeamMembership,
)
from app.services import oe_yapat_service


logger = logging.getLogger(__name__)


class CustomTaxonomyServiceError(Exception):
    """Base exception for custom taxonomy service errors"""
    pass


def create_conversation(
    user_id: int,
    db: Session,
    team_id: Optional[int] = None,
) -> TaxonomyConversation:
    """
    Create a new taxonomy conversation.

    Args:
        user_id: User creating the conversation
        db: Database session
        team_id: Team the conversation belongs to (None for personal conversations)

    Returns:
        Created TaxonomyConversation
    """
    conversation = TaxonomyConversation(
        user_id=user_id,
        team_id=team_id,
        status=ConversationStatus.IN_PROGRESS
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    logger.info(f"Created conversation {conversation.id} for user {user_id} in team {team_id}")
    return conversation


def add_message(
    conversation_id: int,
    role: MessageRole,
    content: str,
    db: Session,
    metadata: Optional[Dict[str, Any]] = None
) -> TaxonomyMessage:
    """
    Add a message to a conversation.
    
    Args:
        conversation_id: ID of the conversation
        role: Role of the message sender (user, assistant, system)
        content: Message content
        db: Database session
        metadata: Optional metadata for the message
        
    Returns:
        Created TaxonomyMessage
    """
    message = TaxonomyMessage(
        conversation_id=conversation_id,
        role=role.value if isinstance(role, MessageRole) else role,
        content=content,
        message_metadata=metadata
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    
    return message


async def process_user_prompt(
    conversation_id: int,
    prompt: str,
    db: Session
) -> Dict[str, Any]:
    """
    Process user prompt and generate taxonomy using OE_YAPAT service.
    
    Args:
        conversation_id: ID of the conversation
        prompt: User's prompt for taxonomy generation
        db: Database session
        
    Returns:
        Dict containing generated taxonomy and message data
        
    Raises:
        CustomTaxonomyServiceError: If conversation not found or generation fails
    """
    # Validate conversation exists and is in progress
    conversation = db.query(TaxonomyConversation).filter(
        TaxonomyConversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise CustomTaxonomyServiceError(f"Conversation {conversation_id} not found")
    
    if conversation.status != ConversationStatus.IN_PROGRESS:
        raise CustomTaxonomyServiceError(
            f"Conversation {conversation_id} is not in progress (status: {conversation.status})"
        )
    
    # Sanitize prompt
    sanitized_prompt = oe_yapat_service.sanitize_prompt(prompt)
    
    # Add user message
    user_message = add_message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=sanitized_prompt,
        db=db
    )
    
    try:
        # Call OE_YAPAT service with domain context
        system_context = (
            "We are going to annotate an audio dataset in the domain of wildlife monitoring. "
        )
        effective_prompt = f"{system_context}User request: {sanitized_prompt}"
        logger.info(f"Processing prompt for conversation {conversation_id}")
        response = await oe_yapat_service.generate_taxonomy(
            prompt=effective_prompt,
            context={"conversation_id": conversation_id}
        )
        
        # Extract taxonomy data and response text
        taxonomy_data = response.get("taxonomy_data", {})
        metadata = response.get("metadata", {})
        response_text = taxonomy_data.get("response_text") if isinstance(taxonomy_data, dict) else None
        
        # Use LLM's actual response text, or fallback to a generic message
        if not response_text:
            # Fallback message if no response text provided
            if taxonomy_data and taxonomy_data.get("nodes"):
                node_count = len(taxonomy_data.get("nodes", []))
                response_text = f"I've found {node_count} relevant taxonomy entries based on your query. Please review them and let me know if you'd like to add them to your label space or refine the search."
            else:
                response_text = "I couldn't find any matching taxonomy entries for your query. Could you please rephrase or provide more details?"
        
        # Create assistant response message with actual LLM response
        assistant_message = add_message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            db=db,
            metadata={
                "taxonomy_data": taxonomy_data,
                "generation_metadata": metadata
            }
        )
        
        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        db.commit()
        
        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "taxonomy_data": taxonomy_data,
            "metadata": metadata
        }
        
    except oe_yapat_service.OEYapatServiceError as e:
        logger.error(f"OE_YAPAT service error for conversation {conversation_id}: {e}")
        # Add error message
        add_message(
            conversation_id=conversation_id,
            role=MessageRole.SYSTEM,
            content=f"Error generating taxonomy: {str(e)}",
            db=db,
            metadata={"error": True}
        )
        raise  # re-raise the original OEYapatServiceError so the API returns 503


def add_to_label_space(
    conversation_id: int,
    user_id: int,
    db: Session,
    message_id: Optional[int] = None,
    indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Add species from a specific assistant response to the label space list.
    
    Args:
        conversation_id: ID of the conversation
        user_id: ID of the user adding to label space
        db: Database session
        message_id: Optional ID of the specific assistant message to add from.
                   If not provided, uses the last assistant message.
        indices: Optional list of 1-based indices to add specific species.
                If not provided, adds all species from the message.
        
    Returns:
        Dict with conversation, added_items list, and skipped_count
        
    Raises:
        CustomTaxonomyServiceError: If conversation not found or already frozen
    """
    # Get conversation
    conversation = db.query(TaxonomyConversation).filter(
        TaxonomyConversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise CustomTaxonomyServiceError(f"Conversation {conversation_id} not found")
    
    if conversation.is_frozen:
        raise CustomTaxonomyServiceError("Label space is frozen. Cannot add more items.")
    
    if conversation.status != ConversationStatus.IN_PROGRESS:
        raise CustomTaxonomyServiceError("Conversation is not in progress")
    
    # Find the target assistant message
    if message_id:
        # Get specific message by ID
        target_message = db.query(TaxonomyMessage).filter(
            TaxonomyMessage.id == message_id,
            TaxonomyMessage.conversation_id == conversation_id,
            TaxonomyMessage.role == MessageRole.ASSISTANT.value
        ).first()
        
        if not target_message:
            raise CustomTaxonomyServiceError(f"Assistant message {message_id} not found in this conversation")
    else:
        # Get the last assistant message
        target_message = db.query(TaxonomyMessage).filter(
            TaxonomyMessage.conversation_id == conversation_id,
            TaxonomyMessage.role == MessageRole.ASSISTANT.value
        ).order_by(TaxonomyMessage.created_at.desc()).first()
    
    if not target_message or not target_message.message_metadata:
        raise CustomTaxonomyServiceError("No species data found in the specified message")
    
    taxonomy_data = target_message.message_metadata.get("taxonomy_data")
    if not taxonomy_data:
        raise CustomTaxonomyServiceError("No species data found in last message")
    
    # oe_yapat returns taxonomy_data as { "nodes": [ {...}, ... ], "metadata": {...} }
    nodes = taxonomy_data.get("nodes") if isinstance(taxonomy_data.get("nodes"), list) else []
    if not nodes:
        raise CustomTaxonomyServiceError("No species data found in the specified message")
    
    # Initialize label_space if None
    if conversation.label_space is None:
        conversation.label_space = []
    
    # Filter nodes by indices if provided
    if indices is not None:
        # Convert 1-based indices to 0-based and filter valid ones
        selected_nodes = []
        skipped = 0
        for idx in indices:
            zero_based = idx - 1
            if 0 <= zero_based < len(nodes):
                selected_nodes.append(nodes[zero_based])
            else:
                skipped += 1
                logger.warning(f"Index {idx} out of range (1-{len(nodes)})")
        nodes_to_add = selected_nodes
        skipped_count = skipped
    else:
        # Add all nodes
        nodes_to_add = nodes
        skipped_count = 0
    
    if not nodes_to_add:
        raise CustomTaxonomyServiceError("No valid indices provided or all indices were out of range")

    # Check for existing IDs to avoid duplicates
    existing_ids = {item.get("taxon_id") for item in conversation.label_space if item.get("taxon_id")}

    added_items = []
    for node in nodes_to_add:
        # Each node has: id, name, scientific_name, rank, metadata (from oe_yapat)
        species_name = node.get("name") or node.get("canonical_name") or "Unknown"
        scientific_name = node.get("scientific_name")
        taxon_id = node.get("id") or node.get("taxon_id")
        
        # Skip duplicates
        if taxon_id and taxon_id in existing_ids:
            logger.info(f"Skipping duplicate: {species_name} ({taxon_id})")
            skipped_count += 1
            continue
        
        node_meta = node.get("metadata") or {}
        if not isinstance(node_meta, dict):
            node_meta = {}
        item_id = str(uuid.uuid4())
        new_item = {
            "id": item_id,
            "name": species_name,
            "scientific_name": scientific_name,
            "taxon_id": taxon_id,
            "metadata": {
                "rank": node.get("rank"),
                "family": node.get("family") or node_meta.get("family"),
                "kingdom": node.get("kingdom") or node_meta.get("kingdom"),
                **node_meta
            },
            "added_at": datetime.utcnow().isoformat()
        }
        conversation.label_space.append(new_item)
        if taxon_id:
            existing_ids.add(taxon_id)
        added_items.append(new_item)

    if not added_items:
        # All selected items were duplicates: return success so UI keeps label space and can show "Already in label space"
        db.refresh(conversation)
        logger.info(
            "Add to label space: all %d item(s) already in label space (conversation %s)",
            len(nodes_to_add),
            conversation_id,
        )
        return {
            "conversation": conversation,
            "added_items": [],
            "skipped_count": skipped_count,
        }

    # Persist new items and add system message
    flag_modified(conversation, "label_space")
    conversation.updated_at = datetime.utcnow()
    names_added = ", ".join(i["name"] for i in added_items)
    add_message(
        conversation_id=conversation_id,
        role=MessageRole.SYSTEM,
        content=f"✓ Added '{names_added}' to your label space.",
        db=db,
        metadata={"action": "added_to_label_space", "item_ids": [i["id"] for i in added_items]},
    )
    db.commit()
    db.refresh(conversation)

    logger.info(f"Added %d item(s) to label space in conversation {conversation_id}", len(added_items))
    return {
        "conversation": conversation,
        "added_items": added_items,
        "skipped_count": skipped_count,
    }


def reject_taxonomy(conversation_id: int, db: Session) -> TaxonomyConversation:
    """
    Reject/cancel a conversation.
    
    Args:
        conversation_id: ID of the conversation
        db: Database session
        
    Returns:
        Updated conversation
    """
    conversation = db.query(TaxonomyConversation).filter(
        TaxonomyConversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise CustomTaxonomyServiceError(f"Conversation {conversation_id} not found")
    
    conversation.status = ConversationStatus.CANCELLED
    conversation.updated_at = datetime.utcnow()
    
    add_message(
        conversation_id=conversation_id,
        role=MessageRole.SYSTEM,
        content="Conversation cancelled by user.",
        db=db
    )
    
    db.commit()
    db.refresh(conversation)
    
    logger.info(f"Cancelled conversation {conversation_id}")
    return conversation


def get_available_taxonomies(
    team_id: int,
    user_id: int,
    db: Session,
    status: Optional[str] = None
) -> List[CustomTaxonomy]:
    """
    Get all taxonomies available to a user.
    
    Returns:
    - Taxonomies from user's team
    - Global taxonomies (is_global=True)
    
    Args:
        team_id: Team ID
        user_id: User ID
        db: Database session
        status: Optional status filter
        
    Returns:
        List of available CustomTaxonomy objects
    """
    query = db.query(CustomTaxonomy).filter(
        or_(
            CustomTaxonomy.team_id == team_id,
            CustomTaxonomy.is_global == True
        )
    )
    
    if status:
        query = query.filter(CustomTaxonomy.status == status)
    
    taxonomies = query.order_by(CustomTaxonomy.created_at.desc()).all()
    
    return taxonomies


def get_taxonomy_by_id(taxonomy_id: str, db: Session) -> Optional[CustomTaxonomy]:
    """
    Get a custom taxonomy by its taxonomy_id.
    
    Args:
        taxonomy_id: Taxonomy ID (e.g., 'custom:uuid')
        db: Database session
        
    Returns:
        CustomTaxonomy or None if not found
    """
    return db.query(CustomTaxonomy).filter(
        CustomTaxonomy.taxonomy_id == taxonomy_id
    ).first()


def resolve_custom_taxon_id(
    taxon_id: str,
    taxonomy_id: str,
    db: Session
) -> Optional[Dict[str, Any]]:
    """
    Resolve a taxon ID within a custom taxonomy.
    
    Args:
        taxon_id: Taxon ID to resolve
        taxonomy_id: Taxonomy ID containing the taxon
        db: Database session
        
    Returns:
        Dict with taxon details or None if not found
    """
    taxonomy = get_taxonomy_by_id(taxonomy_id, db)
    if not taxonomy:
        return None
    
    # Search for taxon in taxonomy_data
    taxonomy_data = taxonomy.taxonomy_data
    
    # Recursive search function
    def find_taxon(data: Dict[str, Any], target_id: str) -> Optional[Dict[str, Any]]:
        # Check if data has nodes
        if "nodes" in data:
            for node in data["nodes"]:
                if node.get("id") == target_id:
                    return node
                # Check children
                if "children" in node:
                    result = find_taxon({"nodes": node["children"]}, target_id)
                    if result:
                        return result
        return None
    
    taxon_node = find_taxon(taxonomy_data, taxon_id)
    
    if taxon_node:
        return {
            "taxon_id": f"{taxonomy_id}:{taxon_id}",
            "name": taxon_node.get("name"),
            "rank": taxon_node.get("rank"),
            "parent_id": taxon_node.get("parent_id"),
            "taxonomy_id": taxonomy_id,
            "taxonomy_name": taxonomy.name,
            "metadata": taxon_node.get("metadata")
        }
    
    return None


def get_conversation_by_id(conversation_id: int, db: Session) -> Optional[TaxonomyConversation]:
    """
    Get a conversation by ID with all messages loaded.
    
    Args:
        conversation_id: Conversation ID
        db: Database session
        
    Returns:
        TaxonomyConversation with messages or None
    """
    return db.query(TaxonomyConversation).filter(
        TaxonomyConversation.id == conversation_id
    ).first()


def update_taxonomy(
    taxonomy_db_id: int,
    db: Session,
    name: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None
) -> CustomTaxonomy:
    """
    Update a custom taxonomy.
    
    Args:
        taxonomy_db_id: Database ID of the taxonomy
        db: Database session
        name: Optional new name
        description: Optional new description
        status: Optional new status
        
    Returns:
        Updated CustomTaxonomy
    """
    taxonomy = db.query(CustomTaxonomy).filter(CustomTaxonomy.id == taxonomy_db_id).first()
    
    if not taxonomy:
        raise CustomTaxonomyServiceError(f"Taxonomy {taxonomy_db_id} not found")
    
    if name is not None:
        taxonomy.name = name
    if description is not None:
        taxonomy.description = description
    if status is not None:
        taxonomy.status = status
    
    taxonomy.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(taxonomy)
    
    logger.info(f"Updated taxonomy {taxonomy.taxonomy_id}")
    return taxonomy


def archive_taxonomy(taxonomy_db_id: int, db: Session) -> CustomTaxonomy:
    """
    Archive a custom taxonomy (soft delete).
    
    Args:
        taxonomy_db_id: Database ID of the taxonomy
        db: Database session
        
    Returns:
        Archived CustomTaxonomy
    """
    return update_taxonomy(taxonomy_db_id, db, status=TaxonomyStatus.ARCHIVED)
