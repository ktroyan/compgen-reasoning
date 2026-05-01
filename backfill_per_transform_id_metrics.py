"""
backfill_per_transform_id_metrics.py

Retroactively computes and uploads per-transformation ID metrics to WandB
for runs completed before this feature was implemented.

For each run missing test/id_per_transform_grid_acc_std:
  1. Download test_predictions.json from WandB.
  2. Reconstruct per-transformation ID metrics from the per-sample data.
  3. Compute aggregate scalars (std, worst-10 avg, best-10 avg per metric).
  4. Update the run summary with those scalars.

Note: WandB Tables and bar charts cannot be retroactively added via the API.
Only the scalar summary columns are backfilled here. New runs will have the
full Table and chart panels automatically via on_test_epoch_end.

Usage:
  python backfill_per_transform_id_metrics.py [--entity ENTITY] [--project PROJECT] [--dry-run]
"""

import argparse
import json
import sys
import tempfile

import wandb


SENTINEL_KEY = "test/id_per_transform_grid_acc_std"
METRIC_KEYS  = ["acc", "grid_acc", "acc_no_pad", "grid_acc_no_pad", "obj_acc"]


def _compute_per_transform_id(samples: list) -> dict:
    """
    Group ID samples from test_predictions.json by transformation suite and
    compute average metrics per group.

    Returns {} if there are no ID samples or if the required per-sample metric
    fields are absent (runs that predate their introduction).
    """
    id_samples = [o for o in samples if o.get("domain_type") == "id"]
    if not id_samples:
        return {}

    # Verify the required fields exist on the first sample
    required = {f"id_{m}" for m in METRIC_KEYS}
    if not required.issubset(id_samples[0].keys()):
        return {}

    groups: dict = {}
    for o in id_samples:
        key = "|".join(o.get("transformation_suite", []))
        groups.setdefault(key, []).append(o)

    result = {}
    for key, group in groups.items():
        n = len(group)
        entry: dict = {"n_samples": n}
        for m in METRIC_KEYS:
            entry[f"id_{m}"] = sum(o[f"id_{m}"] for o in group) / n
        result[key] = entry
    return result


def _compute_scalars(per_transform_id: dict) -> dict:
    """Compute std, worst-10 avg, best-10 avg for each metric across all transformation types."""
    scalars = {}
    for m in METRIC_KEYS:
        vals = sorted(entry[f"id_{m}"] for entry in per_transform_id.values())
        n = len(vals)
        mean_val = sum(vals) / n
        std       = (sum((v - mean_val) ** 2 for v in vals) / n) ** 0.5
        k         = min(10, n)
        worst10   = sum(vals[:k]) / k
        best10    = sum(vals[-k:]) / k
        scalars[f"test/id_per_transform_{m}_std"]         = std
        scalars[f"test/id_per_transform_{m}_worst10_avg"] = worst10
        scalars[f"test/id_per_transform_{m}_best10_avg"]  = best10
    return scalars


def _fetch_samples(run) -> list | None:
    """Download test_predictions.json and return the samples list, or None on failure."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = run.file("test_predictions.json").download(root=tmpdir, replace=True)
            content = downloaded.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            return json.loads(content).get("samples", [])
    except Exception as e:
        print(f"  [WARN] Could not download test_predictions.json for {run.id}: {e}")
        return None


def backfill(entity: str, project: str, dry_run: bool) -> None:
    api = wandb.Api()
    print(f"Fetching all runs from WandB project {entity}/{project}...")
    try:
        runs = list(api.runs(f"{entity}/{project}"))
    except Exception as e:
        print(f"Error fetching runs: {e}")
        sys.exit(1)
    print(f"  {len(runs)} run(s) found.\n")

    updated               = 0
    skipped_already_done  = 0
    skipped_no_data       = 0
    errors                = 0

    for run in runs:
        run_path      = f"{entity}/{project}/{run.id}"
        existing_keys = set(run.summary.keys())

        if SENTINEL_KEY in existing_keys:
            skipped_already_done += 1
            continue

        samples = _fetch_samples(run)
        if samples is None:
            skipped_no_data += 1
            continue

        per_transform_id = _compute_per_transform_id(samples)
        if not per_transform_id:
            print(f"[SKIP – no ID samples or fields missing] {run_path}")
            skipped_no_data += 1
            continue

        scalars        = _compute_scalars(per_transform_id)
        new_metrics    = {k: v for k, v in scalars.items() if k not in existing_keys}
        already_present = {k for k in scalars if k in existing_keys}

        if already_present:
            print(
                f"[NOTE] {run_path}: {len(already_present)} key(s) already present "
                f"(not overwriting): {sorted(already_present)}"
            )

        if not new_metrics:
            print(f"[SKIP – all keys already present] {run_path}")
            continue

        n_types = len(per_transform_id)
        action  = "[DRY-RUN] Would update" if dry_run else "[UPDATE]"
        print(f"{action} {run_path}  ({n_types} transformation type(s))")
        for k, v in sorted(new_metrics.items()):
            print(f"    {k} = {v:.6f}")

        if not dry_run:
            try:
                run.summary.update(new_metrics)
                run.update()
                updated += 1
            except Exception as e:
                print(f"  [ERROR – update failed]: {e}")
                errors += 1

    print("\n--- Summary ---")
    if dry_run:
        print("Dry-run mode: no changes were made to WandB.")
    else:
        print(f"Runs updated:                {updated}")
    print(f"Runs already done (skipped): {skipped_already_done}")
    print(f"Runs with no usable data:    {skipped_no_data}")
    print(f"Errors:                      {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill per-transformation ID metrics to WandB run summaries."
    )
    parser.add_argument("--entity",  default="VisReas-ETHZ",      help="WandB entity name")
    parser.add_argument("--project", default="compgen-reasoning", help="WandB project name")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be uploaded without making any WandB changes.",
    )
    args = parser.parse_args()
    backfill(args.entity, args.project, dry_run=args.dry_run)
