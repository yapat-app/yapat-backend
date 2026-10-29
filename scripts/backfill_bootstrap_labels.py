"""
One-off repair of label storage for datasets whose first model was bootstrapped.

Before the fix in PAMActiveLearningService._submit_bootstrap_feedback /
train_from_scratch, bootstrap training (first model trained from users' own
labels, no metadata CSV) left two problems:

1. The labels were stored only in al_snippet_annotation (USER), never in the
   canonical `annotations` table, so the dataset card showed 0 annotated
   snippets and the export stayed disabled.
2. The training copied those labels in again as GROUND_TRUTH rows tied to the
   bootstrap checkpoint, so a later REJECT left the label in training data.

This script, per bootstrap checkpoint (training_mode cold_start, no
metadata_path):
* deletes the GROUND_TRUTH rows written by that checkpoint, and
* writes canonical annotations for the snippets labelled through bootstrap
  feedback, from their *current* USER labels, when the user has none yet.

Without --apply this is a dry run that only reports. Run inside the API
container and take a database backup before using --apply:

    docker compose ... exec api python scripts/backfill_bootstrap_labels.py
    docker compose ... exec api python scripts/backfill_bootstrap_labels.py --dataset-id 39 --apply
"""

import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.database import SessionLocal  # noqa: E402
from app.models.annotation import Annotation  # noqa: E402
from app.models.dataset import Dataset  # noqa: E402
from app.models.pam_active_learning import (  # noqa: E402
    ALAnnotationSource,
    ALFeedbackEvent,
    ALModelCheckpoint,
    ALSnippetAnnotation,
)
from app.services.pam_al.service import PAMActiveLearningService  # noqa: E402


def bootstrap_checkpoints(db, dataset_id=None):
    query = db.query(ALModelCheckpoint)
    if dataset_id is not None:
        query = query.filter(ALModelCheckpoint.dataset_id == dataset_id)
    return [
        ckpt for ckpt in query.order_by(ALModelCheckpoint.id).all()
        if (ckpt.hyperparameters or {}).get("training_mode") == "cold_start"
        and not (ckpt.hyperparameters or {}).get("metadata_path")
    ]


def missing_canonical(db, dataset_id):
    """{(snippet_id, user_id): [labels]} for bootstrap-labelled snippets with no canonical row."""
    pairs = {
        (snippet_id, user_id)
        for snippet_id, user_id in db.query(ALFeedbackEvent.snippet_id, ALFeedbackEvent.user_id)
        .filter(ALFeedbackEvent.dataset_id == dataset_id, ALFeedbackEvent.model_checkpoint_id.is_(None))
        .distinct()
        if user_id is not None
    }
    labels = defaultdict(list)
    for snippet_id, user_id, label in db.query(
        ALSnippetAnnotation.snippet_id, ALSnippetAnnotation.user_id, ALSnippetAnnotation.label
    ).filter(
        ALSnippetAnnotation.dataset_id == dataset_id,
        ALSnippetAnnotation.source == ALAnnotationSource.USER,
    ):
        if (snippet_id, user_id) in pairs:
            labels[(snippet_id, user_id)].append(label)

    out = {}
    for (snippet_id, user_id), names in labels.items():
        has_canonical = (
            db.query(Annotation.id)
            .filter(Annotation.snippet_id == snippet_id, Annotation.user_id == user_id)
            .first()
        )
        if has_canonical is None:
            out[(snippet_id, user_id)] = sorted(set(names))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair label storage of bootstrapped datasets.")
    parser.add_argument("--apply", action="store_true", help="Write changes. Without it, only report.")
    parser.add_argument("--dataset-id", type=int, help="Limit the repair to one dataset.")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        names = dict(db.query(Dataset.id, Dataset.name).all())
        checkpoints = bootstrap_checkpoints(db, args.dataset_id)
        if not checkpoints:
            print("No bootstrap checkpoints found.")
            return 0

        plans = []
        for dataset_id in sorted({c.dataset_id for c in checkpoints}):
            ckpt_ids = [c.id for c in checkpoints if c.dataset_id == dataset_id]
            gt_copies = (
                db.query(ALSnippetAnnotation)
                .filter(
                    ALSnippetAnnotation.dataset_id == dataset_id,
                    ALSnippetAnnotation.source == ALAnnotationSource.GROUND_TRUTH,
                    ALSnippetAnnotation.model_checkpoint_id.in_(ckpt_ids),
                )
                .count()
            )
            missing = missing_canonical(db, dataset_id)
            plans.append((dataset_id, ckpt_ids, gt_copies, missing))
            print(
                f"dataset {dataset_id} {names.get(dataset_id, '?')}: bootstrap checkpoints {ckpt_ids}, "
                f"GROUND_TRUTH copies to delete {gt_copies}, "
                f"snippets missing canonical annotations {len(missing)}"
            )
        db.commit()   # end the read transaction before any GBIF lookups

        if not args.apply:
            print("\nDry run: nothing changed. Re-run with --apply to write.")
            return 0

        svc = PAMActiveLearningService(db)
        for dataset_id, ckpt_ids, _, missing in plans:
            deleted = (
                db.query(ALSnippetAnnotation)
                .filter(
                    ALSnippetAnnotation.dataset_id == dataset_id,
                    ALSnippetAnnotation.source == ALAnnotationSource.GROUND_TRUTH,
                    ALSnippetAnnotation.model_checkpoint_id.in_(ckpt_ids),
                )
                .delete(synchronize_session=False)
            )
            for (snippet_id, user_id), labels in missing.items():
                svc._try_store_final_annotations(
                    dataset_id=dataset_id, snippet_id=snippet_id, labels=labels, user_id=user_id,
                )
            db.commit()
            print(
                f"dataset {dataset_id}: deleted {deleted} GROUND_TRUTH copies, "
                f"wrote canonical annotations for {len(missing)} snippets"
            )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
