"""
Custom Taxonomy API endpoints

Provides chatbot interface for taxonomy generation and management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from typing import List, Optional

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.models.team import Team, TeamMembership
from app.core.permissions import require_team_member, check_team_member, check_admin
from app.schemas.custom_taxonomy import (
    CustomTaxonomyResponse,
    CustomTaxonomyUpdate,
    CustomTaxonomyListResponse,
)
from app.schemas.taxonomy_conversation import (
    ConversationCreate,
    ConversationResponse,
    ChatRequest,
    ChatResponse,
    AddToLabelSpaceRequest,
    AddToLabelSpaceResponse,
    FreezeLabelSpaceRequest,
    FreezeLabelSpaceResponse,
    MessageResponse,
    ConversationListResponse,
    LabelSpaceResponse,
    LabelSpaceItem,
)
from app.services import custom_taxonomy_service
from app.services.custom_taxonomy_service import CustomTaxonomyServiceError
from app.services.oe_yapat_service import OEYapatServiceError


router = APIRouter()


def _conversation_response_slim_message(conversation, message_id_to_slim: int) -> ConversationResponse:
    """
    Build ConversationResponse from conversation; for the message with the given id,
    omit taxonomy_data from metadata so it only appears in the top-level message (no duplication).
    """
    conv = ConversationResponse.model_validate(conversation)
    if not conv.messages:
        return conv
    new_messages = []
    for m in conv.messages:
        if m.id == message_id_to_slim and getattr(m, "metadata", None):
            meta = m.metadata or {}
            slim_meta = {k: v for k, v in meta.items() if k != "taxonomy_data"}
            if "taxonomy_data" in meta:
                slim_meta["_taxonomy_in_response_message"] = True
            new_messages.append(MessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                metadata=slim_meta,
            ))
        else:
            new_messages.append(m)
    return conv.model_copy(update={"messages": new_messages})


# ============================================================================
# CHATBOT / CONVERSATION ENDPOINTS
# ============================================================================

@router.post("/chat/start", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def start_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Start a new taxonomy generation conversation.
    
    Creates a new conversation context for generating a custom taxonomy.
    User must be a member of the specified team.
    """
    # Verify team exists
    team = db.query(Team).filter(Team.id == request.team_id).first()
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Team {request.team_id} not found"
        )
    # Verify user is member of the team
    require_team_member(current_user, request.team_id, db)
    
    try:
        conversation = custom_taxonomy_service.create_conversation(
            user_id=current_user.id,
            team_id=request.team_id,
            db=db
        )
        return conversation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.post("/chat/{conversation_id}/message", response_model=ChatResponse)
async def send_message(
    conversation_id: int,
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Send a message in a conversation to generate or refine taxonomy.
    
    The OE_YAPAT service will process the prompt and return a generated taxonomy.
    User can send multiple messages to refine the taxonomy before accepting.
    """
    # Get conversation and verify access
    conversation = custom_taxonomy_service.get_conversation_by_id(conversation_id, db)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )
    
    # Verify user is member of the conversation's team
    require_team_member(current_user, conversation.team_id, db)
    
    try:
        # Process the prompt and generate taxonomy
        result = await custom_taxonomy_service.process_user_prompt(
            conversation_id=conversation_id,
            prompt=request.prompt,
            db=db
        )
        
        # Refresh conversation to get updated messages
        db.refresh(conversation)
        
        # Build conversation for response: omit full taxonomy_data from the message we're returning at top level (no duplication)
        conv_data = _conversation_response_slim_message(
            conversation, result["assistant_message"].id
        )
        
        return ChatResponse(
            message=result["assistant_message"],
            conversation=conv_data
        )
        
    except OEYapatServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except CustomTaxonomyServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@router.get("/chat/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get conversation details and message history.
    
    Returns the full conversation including all messages exchanged.
    """
    conversation = custom_taxonomy_service.get_conversation_by_id(conversation_id, db)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )
    
    # Verify user is member of the conversation's team
    require_team_member(current_user, conversation.team_id, db)
    
    return conversation


@router.get("/chat/{conversation_id}/label-space", response_model=LabelSpaceResponse)
def get_label_space(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get label space items for a conversation.
    
    Returns only the label space items without the full conversation data.
    This is a convenient endpoint to fetch just the items added to the label space.
    """
    conversation = custom_taxonomy_service.get_conversation_by_id(conversation_id, db)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )
    
    # Verify user is member of the conversation's team
    require_team_member(current_user, conversation.team_id, db)
    
    # Convert label_space from dict/list to LabelSpaceItem objects using Pydantic validation
    label_space_items = []
    if conversation.label_space:
        for item in conversation.label_space:
            # Pydantic will validate and convert dict to LabelSpaceItem
            label_space_items.append(LabelSpaceItem.model_validate(item))
    
    return LabelSpaceResponse(
        conversation_id=conversation.id,
        is_frozen=conversation.is_frozen,
        items=label_space_items,
        total=len(label_space_items)
    )


@router.post("/chat/{conversation_id}/add", response_model=AddToLabelSpaceResponse)
def add_to_label_space(
    conversation_id: int,
    request: AddToLabelSpaceRequest = AddToLabelSpaceRequest(),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add the last AI-suggested species to the label space list.
    
    This adds the species from the last assistant message to the accumulating
    label space. Users can continue to add more species before freezing.
    """
    # Get conversation and verify access
    conversation = custom_taxonomy_service.get_conversation_by_id(conversation_id, db)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )
    
    # Verify user is member of the conversation's team
    require_team_member(current_user, conversation.team_id, db)
    
    try:
        result = custom_taxonomy_service.add_to_label_space(
            conversation_id=conversation_id,
            user_id=current_user.id,
            db=db,
            message_id=request.message_id,
            indices=request.indices
        )
        
        return AddToLabelSpaceResponse(
            conversation=result["conversation"],
            added_items=result["added_items"],
            skipped_count=result["skipped_count"]
        )
        
    except CustomTaxonomyServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/chat/{conversation_id}/freeze", response_model=FreezeLabelSpaceResponse)
def freeze_label_space(
    conversation_id: int,
    request: FreezeLabelSpaceRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Freeze the label space and create a custom taxonomy.
    
    This finalizes the label space session and creates a custom taxonomy
    from all accumulated species. The taxonomy becomes available for
    annotation by all team members. No more items can be added after freezing.
    """
    # Get conversation and verify access
    conversation = custom_taxonomy_service.get_conversation_by_id(conversation_id, db)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )
    
    # Verify user is member of the conversation's team
    require_team_member(current_user, conversation.team_id, db)
    
    try:
        from app.services.custom_taxonomy_service_freeze import freeze_label_space as freeze_func
        
        result = freeze_func(
            conversation_id=conversation_id,
            user_id=current_user.id,
            name=request.name,
            db=db,
            description=request.description
        )
        
        return FreezeLabelSpaceResponse(
            conversation=result["conversation"],
            taxonomy=CustomTaxonomyResponse.model_validate(result["taxonomy"])
        )
        
    except CustomTaxonomyServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/chat/{conversation_id}/item/{item_id}", response_model=ConversationResponse)
def remove_from_label_space(
    conversation_id: int,
    item_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Remove a specific item from the label space list.
    
    Allows users to remove an item they added by mistake before freezing.
    """
    # Get conversation and verify access
    conversation = custom_taxonomy_service.get_conversation_by_id(conversation_id, db)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )
    
    # Verify user is member of the conversation's team
    require_team_member(current_user, conversation.team_id, db)
    
    if conversation.is_frozen:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove items from a frozen label space"
        )
    
    # Remove item from label_space
    if not conversation.label_space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Label space is empty"
        )
    
    original_length = len(conversation.label_space)
    conversation.label_space = [item for item in conversation.label_space if item.get("id") != item_id]
    flag_modified(conversation, "label_space")
    
    if len(conversation.label_space) == original_length:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found in label space"
        )
    
    db.commit()
    db.refresh(conversation)
    
    return conversation


@router.post("/chat/{conversation_id}/cancel", response_model=ConversationResponse)
def cancel_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Cancel the conversation without creating a taxonomy.
    
    The conversation will be marked as cancelled and the label space will be discarded.
    """
    # Get conversation and verify access
    conversation = custom_taxonomy_service.get_conversation_by_id(conversation_id, db)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )
    
    # Verify user is member of the conversation's team
    require_team_member(current_user, conversation.team_id, db)
    
    try:
        conversation = custom_taxonomy_service.reject_taxonomy(conversation_id, db)
        return conversation
    except CustomTaxonomyServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ============================================================================
# TAXONOMY MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/custom", response_model=CustomTaxonomyListResponse)
def list_custom_taxonomies(
    team_id: Optional[int] = Query(None, description="Filter by team ID"),
    status: Optional[str] = Query(None, description="Filter by status (draft, active, archived)"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List all custom taxonomies available to the user.
    
    Returns taxonomies from:
    - User's teams (if team_id specified, only that team)
    - Global taxonomies (created by admins)
    """
    # If team_id specified, verify user is member
    if team_id:
        require_team_member(current_user, team_id, db)
        taxonomies = custom_taxonomy_service.get_available_taxonomies(
            team_id=team_id,
            user_id=current_user.id,
            db=db,
            status=status
        )
    else:
        # Get all taxonomies from all user's teams
        # Get user's teams
        memberships = db.query(TeamMembership).filter(
            TeamMembership.user_id == current_user.id
        ).all()
        
        all_taxonomies = []
        for membership in memberships:
            team_taxonomies = custom_taxonomy_service.get_available_taxonomies(
                team_id=membership.team_id,
                user_id=current_user.id,
                db=db,
                status=status
            )
            all_taxonomies.extend(team_taxonomies)
        
        # Remove duplicates (global taxonomies might appear multiple times)
        seen_ids = set()
        taxonomies = []
        for t in all_taxonomies:
            if t.id not in seen_ids:
                taxonomies.append(t)
                seen_ids.add(t.id)
    
    return CustomTaxonomyListResponse(
        taxonomies=taxonomies,
        total=len(taxonomies)
    )


@router.get("/custom/{taxonomy_id}", response_model=CustomTaxonomyResponse)
def get_taxonomy(
    taxonomy_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a custom taxonomy.
    
    Returns the full taxonomy including hierarchical structure.
    User must be a member of the taxonomy's team or it must be a global taxonomy.
    """
    taxonomy = custom_taxonomy_service.get_taxonomy_by_id(taxonomy_id, db)
    if not taxonomy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Taxonomy {taxonomy_id} not found"
        )
    
    # Check access: must be team member or global taxonomy
    if not taxonomy.is_global:
        if not check_team_member(current_user, taxonomy.team_id, db) and not check_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this taxonomy"
            )
    
    return taxonomy


@router.put("/custom/{taxonomy_db_id}", response_model=CustomTaxonomyResponse)
def update_taxonomy(
    taxonomy_db_id: int,
    request: CustomTaxonomyUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update a custom taxonomy's metadata.
    
    Only the creator or team owner can update a taxonomy.
    Can update name, description, and status.
    """
    taxonomy = db.query(custom_taxonomy_service.CustomTaxonomy).filter(
        custom_taxonomy_service.CustomTaxonomy.id == taxonomy_db_id
    ).first()
    
    if not taxonomy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Taxonomy {taxonomy_db_id} not found"
        )
    
    # Check permissions: creator, team owner, or admin
    from app.core.permissions import check_team_owner_membership
    
    is_creator = taxonomy.created_by_user_id == current_user.id
    is_team_owner = check_team_owner_membership(current_user, taxonomy.team_id, db)
    is_admin = check_admin(current_user)
    
    if not (is_creator or is_team_owner or is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the creator, team owner, or admin can update this taxonomy"
        )
    
    try:
        updated_taxonomy = custom_taxonomy_service.update_taxonomy(
            taxonomy_db_id=taxonomy_db_id,
            db=db,
            name=request.name,
            description=request.description,
            status=request.status
        )
        return updated_taxonomy
    except CustomTaxonomyServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/custom/{taxonomy_db_id}", status_code=status.HTTP_204_NO_CONTENT)
def archive_taxonomy(
    taxonomy_db_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Archive a custom taxonomy (soft delete).
    
    Only the creator, team owner, or admin can archive a taxonomy.
    Archived taxonomies are hidden from listings but existing annotations remain valid.
    """
    taxonomy = db.query(custom_taxonomy_service.CustomTaxonomy).filter(
        custom_taxonomy_service.CustomTaxonomy.id == taxonomy_db_id
    ).first()
    
    if not taxonomy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Taxonomy {taxonomy_db_id} not found"
        )
    
    # Check permissions: creator, team owner, or admin
    from app.core.permissions import check_team_owner_membership
    
    is_creator = taxonomy.created_by_user_id == current_user.id
    is_team_owner = check_team_owner_membership(current_user, taxonomy.team_id, db)
    is_admin = check_admin(current_user)
    
    if not (is_creator or is_team_owner or is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the creator, team owner, or admin can archive this taxonomy"
        )
    
    try:
        custom_taxonomy_service.archive_taxonomy(taxonomy_db_id, db)
        return None
    except CustomTaxonomyServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
