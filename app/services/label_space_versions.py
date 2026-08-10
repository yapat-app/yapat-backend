"""
Team label-space versioning.

A team's label space is versioned as CustomTaxonomy rows:

  - A member finalizes their edited label space -> submit_label_space() creates a
    CustomTaxonomy with status=SUBMITTED, auto-named "Version N" (unique per team)
    and attributed to the member via created_by_user_id.
  - The team owner promotes one submitted version -> promote_label_space() sets
    teams.active_custom_taxonomy_id, marks that version ACTIVE, and demotes the
    previously-active version back to SUBMITTED (so it stays re-promotable).

Only the ACTIVE version is usable for annotation (annotation requires
CustomTaxonomy.status == ACTIVE). All team members read the active version via
teams.active_custom_taxonomy_id.
"""

import logging
import re
import secrets
from datetime import datetime
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    CustomTaxonomy,
    TaxonomyConversation,
    ConversationStatus,
    MessageRole,
    TaxonomyStatus,
)
from app.models.team import Team
from app.services.custom_taxonomy_service import CustomTaxonomyServiceError, add_message


logger = logging.getLogger(__name__)

_VERSION_NAME_RE = re.compile(r"^Version\s+(\d+)$")


def _next_version_name(team_id: Optional[int], dataset_id: Optional[int], db: Session) -> str:
    """Return the next "Version N" name, scoped per (team_id, dataset_id)."""
    names = (
        db.query(CustomTaxonomy.name)
        .filter(
            CustomTaxonomy.team_id == team_id,
            CustomTaxonomy.dataset_id == dataset_id,
        )
        .all()
    )
    highest = 0
    for (name,) in names:
        m = _VERSION_NAME_RE.match(name or "")
        if m:
            highest = max(highest, int(m.group(1)))
    return f"Version {highest + 1}"


def _generate_taxonomy_id(db: Session) -> str:
    """Generate a unique custom taxonomy_id string."""
    for attempt in range(10):
        candidate = f"custom:{secrets.token_hex(4)}"
        if not db.query(CustomTaxonomy).filter(CustomTaxonomy.taxonomy_id == candidate).first():
            return candidate
    return f"custom:{secrets.token_hex(8)}"


def submit_label_space(
    conversation_id: int,
    user_id: int,
    db: Session,
    description: Optional[str] = None,
) -> dict:
    """
    Finalize a conversation's label space into a submitted CustomTaxonomy version.

    The version is auto-named "Version N" (unique per team) and attributed to the
    submitting user. It is NOT yet usable for annotation; the team owner must
    promote it (promote_label_space) to make it the team's active version.

    Returns dict with "conversation" and "taxonomy".
    """
    conversation = db.query(TaxonomyConversation).filter(
        TaxonomyConversation.id == conversation_id
    ).first()

    if not conversation:
        raise CustomTaxonomyServiceError(f"Conversation {conversation_id} not found")
    if conversation.is_frozen:
        raise CustomTaxonomyServiceError("Label space is already finalized")
    if conversation.status != ConversationStatus.IN_PROGRESS:
        raise CustomTaxonomyServiceError("Conversation is not in progress")
    if conversation.team_id is None and conversation.dataset_id is None:
        raise CustomTaxonomyServiceError(
            "This conversation has neither a team nor a dataset. Start it with a "
            "team_id or dataset_id before finalizing."
        )
    if not conversation.label_space or len(conversation.label_space) == 0:
        raise CustomTaxonomyServiceError(
            "Label space is empty. Add at least one species before finalizing."
        )

    name = _next_version_name(conversation.team_id, conversation.dataset_id, db)
    taxonomy_id = _generate_taxonomy_id(db)

    taxonomy_data = {
        "nodes": conversation.label_space,
        "metadata": {
            "created_from_conversation": conversation_id,
            "total_species": len(conversation.label_space),
            "created_at": datetime.utcnow().isoformat(),
        },
    }

    custom_taxonomy = CustomTaxonomy(
        taxonomy_id=taxonomy_id,
        team_id=conversation.team_id,
        dataset_id=conversation.dataset_id,
        created_by_user_id=user_id,
        name=name,
        description=description,
        taxonomy_data=taxonomy_data,
        status=TaxonomyStatus.SUBMITTED,
        is_global=False,
        is_label_space_version=True,
    )
    db.add(custom_taxonomy)

    conversation.is_frozen = True
    conversation.status = ConversationStatus.COMPLETED
    conversation.custom_taxonomy_id = custom_taxonomy.id
    conversation.updated_at = datetime.utcnow()

    add_message(
        conversation_id=conversation_id,
        role=MessageRole.SYSTEM,
        content=(
            f"📤 Label space finalized as '{name}' ({len(conversation.label_space)} species) "
            f"and submitted to the team owner for review."
        ),
        db=db,
        metadata={"action": "submitted", "taxonomy_id": taxonomy_id, "version_name": name},
    )

    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise CustomTaxonomyServiceError(
            "Failed to finalize label space (possible duplicate version name). Please retry."
        ) from e

    db.refresh(custom_taxonomy)
    db.refresh(conversation)

    # Mirror the finalized label space into the dataset's quick_labels so the
    # custom labels show up in the dataset card (GET /api/datasets/{id}/quick-labels)
    # for annotation — matching the legacy freeze behavior. Append-and-dedup, so
    # it never removes the model-checkpoint defaults. Best-effort: a failure here
    # must not fail the submit (the version is already committed).
    if conversation.dataset_id:
        from app.services.custom_taxonomy_service import sync_items_to_dataset_quick_labels
        try:
            sync_items_to_dataset_quick_labels(
                conversation.dataset_id,
                list(conversation.label_space or []),
                db,
            )
            db.refresh(conversation)
        except Exception:
            logger.exception(
                "Failed to sync submitted label space to dataset %s quick_labels (conversation %s)",
                conversation.dataset_id,
                conversation_id,
            )
            db.rollback()

    logger.info(
        "Submitted label space version %s (%s) for team %s from conversation %s",
        name, taxonomy_id, conversation.team_id, conversation_id,
    )
    return {"conversation": conversation, "taxonomy": custom_taxonomy}


def promote_label_space(
    team_id: int,
    taxonomy_db_id: int,
    db: Session,
) -> CustomTaxonomy:
    """
    Promote a submitted version to be the team's active label space.

    Sets teams.active_custom_taxonomy_id, marks the promoted version ACTIVE, and
    demotes any previously-active version back to SUBMITTED. Idempotent when the
    target is already the active version.

    Returns the now-active CustomTaxonomy.
    """
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise CustomTaxonomyServiceError(f"Team {team_id} not found")

    target = db.query(CustomTaxonomy).filter(
        CustomTaxonomy.id == taxonomy_db_id
    ).first()
    if not target:
        raise CustomTaxonomyServiceError(f"Label space version {taxonomy_db_id} not found")
    if target.team_id != team_id:
        raise CustomTaxonomyServiceError("This label space version does not belong to the team")
    if not target.is_label_space_version:
        raise CustomTaxonomyServiceError(
            "This taxonomy is not a promotable label-space version."
        )

    # Demote the previously-active version (if any and different) back to SUBMITTED.
    if team.active_custom_taxonomy_id and team.active_custom_taxonomy_id != target.id:
        previous = db.query(CustomTaxonomy).filter(
            CustomTaxonomy.id == team.active_custom_taxonomy_id
        ).first()
        if previous and previous.status == TaxonomyStatus.ACTIVE:
            previous.status = TaxonomyStatus.SUBMITTED

    target.status = TaxonomyStatus.ACTIVE
    team.active_custom_taxonomy_id = target.id

    db.commit()
    db.refresh(target)

    logger.info(
        "Promoted label space version %s ('%s') to active for team %s",
        target.id, target.name, team_id,
    )
    return target
