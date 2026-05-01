"""
backfill_wandb_metrics.py

Retroactively uploads stubbornness, compositional gap, and TRM-specific metrics
to WandB for runs completed before these metrics were logged.

Strategy:
  1. Query WandB for every run in the project.
  2. Skip runs that already have test/stubbornness_acc in their summary.
  3. For each run that needs backfilling:
       a. Look for a matching local .out SLURM log (fast, no network).
       b. If not found locally (file was deleted), download output.log from WandB.
  4. Parse the log, then update the run summary — only adding keys that are
     not already present (existing values are NEVER overwritten).

Safety:
  - Existing summary keys are NEVER overwritten.
  - Use --dry-run to preview all changes without touching WandB.
  - Every update is printed for auditing.

Usage:
  python backfill_wandb_metrics.py [--entity ENTITY] [--project PROJECT]
                                   [--logs-dir logs/] [--dry-run]
"""

import argparse
import re
import sys
import tempfile
import os
from pathlib import Path

import wandb


# ---------------------------------------------------------------------------
# Regex patterns — match the exact log format from on_test_epoch_end
# ---------------------------------------------------------------------------

RE_RUN_URL = re.compile(
    r"WandB run URL: https://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>\S+)"
)
RE_STUBBORNNESS = re.compile(r"\| Stubbornness: (.+)$")
RE_COMP_GAP = re.compile(r"\| Compositional gap: (.+)$")
RE_HALTING = re.compile(r"\| Halting distribution \((\w+)\): mean=([0-9.]+), std=([0-9.]+)")
RE_QHEAD = re.compile(
    r"\| Q-head calibration \((\w+)\): "
    r"overall_precision=([0-9.]+), early_halt_rate=([0-9.]+), early_precision=([0-9.]+)"
)
RE_QHEAD_NO_EARLY = re.compile(r"\| Q-head calibration \((\w+)\): no early halts observed\.")

# Key we use to decide whether a run has already been backfilled / was logged by the fixed code.
SENTINEL_KEY = "test/stubbornness_acc"


def _parse_kv_line(text: str) -> dict:
    """Parse 'key=value, key=value, ...' into {key: float}."""
    result = {}
    for part in text.split(", "):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = float(v.strip())
    return result


def _extract_metrics_from_lines(lines: list[str]) -> dict:
    """Extract all backfill metrics from a list of log lines."""
    metrics: dict[str, float] = {}
    for line in lines:
        m = RE_STUBBORNNESS.search(line)
        if m:
            metrics.update({f"test/{k}": v for k, v in _parse_kv_line(m.group(1)).items()})

        m = RE_COMP_GAP.search(line)
        if m:
            metrics.update({f"test/{k}": v for k, v in _parse_kv_line(m.group(1)).items()})

        m = RE_HALTING.search(line)
        if m:
            pfx = m.group(1)
            metrics[f"test/{pfx}_halt_step_mean"] = float(m.group(2))
            metrics[f"test/{pfx}_halt_step_std"] = float(m.group(3))

        m = RE_QHEAD.search(line)
        if m:
            pfx = m.group(1)
            metrics[f"test/{pfx}_qhead_overall_precision"] = float(m.group(2))
            metrics[f"test/{pfx}_qhead_early_halt_rate"] = float(m.group(3))
            metrics[f"test/{pfx}_qhead_early_precision"] = float(m.group(4))

        m = RE_QHEAD_NO_EARLY.search(line)
        if m:
            metrics[f"test/{m.group(1)}_qhead_early_halt_rate"] = 0.0

    return metrics


def build_local_lookup(logs_dir: Path) -> dict[str, dict]:
    """
    Scan all .out files in logs_dir and return a dict mapping
    run_id -> metrics for every run whose URL appears in a local log.

    Files can contain multiple sequential runs (one URL per run segment).
    """
    lookup: dict[str, dict] = {}
    for log_file in sorted(logs_dir.glob("*.out")):
        lines = log_file.read_text(errors="replace").splitlines()

        # Find where each run starts within this file
        run_starts: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            m = RE_RUN_URL.search(line)
            if m:
                run_starts.append((i, m["run_id"]))

        for idx, (start, run_id) in enumerate(run_starts):
            end = run_starts[idx + 1][0] if idx + 1 < len(run_starts) else len(lines)
            metrics = _extract_metrics_from_lines(lines[start:end])
            lookup[run_id] = metrics   # last write wins if somehow duplicated

    return lookup


def fetch_metrics_from_wandb(run) -> dict:
    """
    Download output.log for a WandB run object and parse metrics from it.
    Returns {} if the file is unavailable or contains no metrics.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = run.file("output.log").download(root=tmpdir, replace=True)
            content = downloaded.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            return _extract_metrics_from_lines(content.splitlines())
    except Exception as e:
        print(f"  [WARN] Could not download output.log for {run.id}: {e}")
        return {}


def backfill(entity: str, project: str, logs_dir: Path, dry_run: bool) -> None:
    # --- Step 1: build local lookup ---
    print(f"Scanning local log files in {logs_dir}...")
    local_lookup = build_local_lookup(logs_dir)
    print(f"  {len(local_lookup)} run(s) found in local log files.\n")

    # --- Step 2: fetch all runs from WandB ---
    api = wandb.Api()
    print(f"Fetching all runs from WandB project {entity}/{project}...")
    try:
        runs = list(api.runs(f"{entity}/{project}"))
    except Exception as e:
        print(f"Error fetching runs: {e}")
        sys.exit(1)
    print(f"  {len(runs)} run(s) found on WandB.\n")

    updated = 0
    skipped_already_done = 0
    skipped_no_metrics = 0
    fetched_from_wandb = 0
    errors = 0

    for run in runs:
        run_path = f"{entity}/{project}/{run.id}"
        existing_keys = set(run.summary.keys())

        # Skip runs that already have stubbornness (fixed code or already backfilled)
        if SENTINEL_KEY in existing_keys:
            skipped_already_done += 1
            continue

        # --- Step 3: get metrics from local file or download from WandB ---
        source = "local"
        if run.id in local_lookup:
            metrics = local_lookup[run.id]
        else:
            print(f"  [WandB download] {run_path} — not in local logs, fetching output.log...")
            metrics = fetch_metrics_from_wandb(run)
            source = "wandb"
            fetched_from_wandb += 1

        if not metrics:
            print(f"[SKIP – no metrics in log] {run_path}  (source: {source})")
            skipped_no_metrics += 1
            continue

        new_metrics = {k: v for k, v in metrics.items() if k not in existing_keys}
        already_present = {k for k in metrics if k in existing_keys}

        if already_present:
            print(
                f"[NOTE] {run_path}: {len(already_present)} key(s) already present "
                f"(not overwriting): {sorted(already_present)}"
            )

        if not new_metrics:
            print(f"[SKIP – all keys already present] {run_path}")
            continue

        action = "[DRY-RUN] Would update" if dry_run else "[UPDATE]"
        print(f"{action} {run_path}  (source: {source})")
        for k, v in sorted(new_metrics.items()):
            print(f"    {k} = {v}")

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
        print(f"Runs updated:                    {updated}")
    print(f"Runs already done (skipped):     {skipped_already_done}")
    print(f"Runs with no metrics in log:     {skipped_no_metrics}")
    print(f"Runs fetched from WandB (no local log): {fetched_from_wandb}")
    print(f"Errors:                          {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill missing metrics to WandB run summaries."
    )
    parser.add_argument("--entity", default="VisReas-ETHZ", help="WandB entity name")
    parser.add_argument("--project", default="compgen-reasoning", help="WandB project name")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing .out SLURM log files (default: logs/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without making any WandB changes.",
    )
    args = parser.parse_args()

    if not args.logs_dir.is_dir():
        print(f"Error: --logs-dir '{args.logs_dir}' is not a directory.")
        sys.exit(1)

    backfill(args.entity, args.project, args.logs_dir, dry_run=args.dry_run)
