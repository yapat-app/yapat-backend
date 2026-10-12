"""
One-off cleanup of AL predictions left behind by superseded checkpoints.

Applies the same retention rule that now runs after every training
(app/services/pam_al/_prediction_retention.py): per dataset and model family,
only the active, previous and first trained checkpoints keep their prediction
rows. Checkpoints, model files, annotations and feedback are not touched.

Without --apply this is a dry run that only reports what would be deleted.
Run inside the API container, where the app and database settings are
available, and take a database backup before using --apply:

    docker compose ... exec api python scripts/prune_al_predictions.py
    docker compose ... exec api python scripts/prune_al_predictions.py --apply
    docker compose ... exec api python scripts/prune_al_predictions.py --dataset-id 62 --apply

Deleted rows are reused by Postgres for new predictions but do not shrink the
table file on disk; reclaiming that space needs a separate VACUUM FULL.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models.dataset import Dataset  # noqa: E402
from app.models.pam_active_learning import ALModelCheckpoint  # noqa: E402
from app.services.pam_al import _prediction_retention as retention  # noqa: E402

# Postgres stores roughly this much per prediction row (measured on staging:
# 28 GB for 17.1M rows); used only to estimate the space a run frees.
APPROX_BYTES_PER_ROW = 1650


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete predictions of superseded AL checkpoints.")
    parser.add_argument("--apply", action="store_true", help="Delete rows. Without it, only report.")
    parser.add_argument("--dataset-id", type=int, help="Limit the cleanup to one dataset.")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        query = db.query(ALModelCheckpoint.dataset_id, ALModelCheckpoint.model_family_name).distinct()
        if args.dataset_id is not None:
            query = query.filter(ALModelCheckpoint.dataset_id == args.dataset_id)
        families = sorted(query.all())
        names = dict(db.query(Dataset.id, Dataset.name).all())

        print(f"{'dataset':<44} {'family':<24} {'keep (active/prev/first)':<26} {'prune':>6} {'rows':>11}")
        total_rows = 0
        plans = []
        for dataset_id, family in families:
            plan = retention.plan_retention(db, dataset_id, family)
            rows = retention.count_predictions(db, plan.prune_ids)
            plans.append(plan)
            total_rows += rows
            keep = f"{plan.active_id}/{plan.previous_id}/{plan.first_trained_id}"
            label = f"{dataset_id} {names.get(dataset_id, '?')}"[:43]
            print(f"{label:<44} {family:<24} {keep:<26} {len(plan.prune_ids):>6} {rows:>11,}")

        print(
            f"\nTotal: {sum(len(p.prune_ids) for p in plans)} checkpoints, {total_rows:,} rows "
            f"(about {total_rows * APPROX_BYTES_PER_ROW / 1e9:.1f} GB)"
        )

        if not args.apply:
            print("\nDry run: nothing deleted. Re-run with --apply to delete.")
            return 0

        deleted = 0
        for plan in plans:
            if plan.prune_ids:
                n = retention.delete_predictions(
                    db, plan.prune_ids, batch_size=settings.PAM_PREDICTION_PRUNE_BATCH_SIZE
                )
                deleted += n
                print(f"dataset {plan.dataset_id} {plan.model_family_name}: deleted {n:,} rows")
        print(f"\nDeleted {deleted:,} prediction rows.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
