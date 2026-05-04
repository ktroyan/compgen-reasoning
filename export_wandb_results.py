"""
export_wandb_results.py

Queries all runs from a WandB project and exports a flat summary to CSV and JSON.
Intended for paper table creation, results analysis, and model comparison.

Usage:
    python export_wandb_results.py                          # all runs
    python export_wandb_results.py --state finished         # only finished runs
    python export_wandb_results.py --runs run_id1 run_id2  # specific runs
    python export_wandb_results.py --artifact-urls          # also fetch artifact URLs (slower)

"""

import argparse
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd
import wandb


# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

CONFIG_FIELDS = {
    # Identity / dataset
    "model.name":                       "model_name",
    "data.data_path":                   "data_path",
    "seed":                             "seed",           # top-level key in config.yaml
}

SUMMARY_FIELDS = [
    # Resource / scale
    "num_trainable_params",
    "num_train_samples",
    # --- ID base metrics ---
    "test/id_acc",
    "test/id_grid_acc",
    "test/id_acc_no_pad",
    "test/id_grid_acc_no_pad",
    "test/id_obj_acc",
    "test/id_loss",
    # --- OOD base metrics ---
    "test/ood_acc",
    "test/ood_grid_acc",
    "test/ood_acc_no_pad",
    "test/ood_grid_acc_no_pad",
    "test/ood_obj_acc",
    "test/ood_loss",
    # ------------------------------
    # Meta-metrics
    # ------------------------------
    # --- Stubbornness ---
    "test/stubbornness_acc",
    "test/stubbornness_grid_acc",
    "test/stubbornness_obj_acc",
    "test/stubbornness_acc_no_pad",
    "test/stubbornness_grid_acc_no_pad",
    # --- Compositional gap ---
    "test/gap_acc",
    "test/gap_grid_acc",
    "test/gap_obj_acc",
    "test/gap_acc_no_pad",
    "test/gap_grid_acc_no_pad",
    # --- Per-transform ID aggregates ---
    "test/id_per_transform_acc_std",
    "test/id_per_transform_acc_worst10_avg",
    "test/id_per_transform_acc_best10_avg",
    "test/id_per_transform_grid_acc_std",
    "test/id_per_transform_grid_acc_worst10_avg",
    "test/id_per_transform_grid_acc_best10_avg",
    "test/id_per_transform_acc_no_pad_std",
    "test/id_per_transform_acc_no_pad_worst10_avg",
    "test/id_per_transform_acc_no_pad_best10_avg",
    "test/id_per_transform_grid_acc_no_pad_std",
    "test/id_per_transform_grid_acc_no_pad_worst10_avg",
    "test/id_per_transform_grid_acc_no_pad_best10_avg",
    "test/id_per_transform_obj_acc_std",
    "test/id_per_transform_obj_acc_worst10_avg",
    "test/id_per_transform_obj_acc_best10_avg",
    # ------------------------------
    # TRM-specific metrics
    # ------------------------------
    # --- TRM halting ---
    "test/id_halt_step_mean",
    "test/id_halt_step_std",
    "test/ood_halt_step_mean",
    "test/ood_halt_step_std",
    # --- TRM Q-head calibration ---
    "test/id_qhead_overall_precision",
    "test/id_qhead_early_halt_rate",
    "test/id_qhead_early_precision",
    "test/ood_qhead_overall_precision",
    "test/ood_qhead_early_halt_rate",
    "test/ood_qhead_early_precision",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(value):
    """Convert NaN / Inf floats to None so JSON serialises cleanly."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _get_nested(cfg_dict, dotted_key, default=None):
    """
    Resolve a dotted key against a WandB config dict.

    WandB stores Hydra configs as a flat dict whose keys contain dots
    (e.g. "model.name", "network.encoder"), and some values are themselves
    dicts (e.g. "network.encoder" -> {"num_heads": 4, "num_layers": 2, ...}).

    Strategy: try progressively longer literal prefixes until one is found
    as a real key, then walk the remaining parts into the value dict.
    Handles all three cases:
      "seed"                       -> cfg["seed"]
      "model.name"                 -> cfg["model.name"]
      "network.encoder.num_layers" -> cfg["network.encoder"]["num_layers"]
    """
    if dotted_key in cfg_dict:
        return cfg_dict[dotted_key]

    parts = dotted_key.split(".")
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in cfg_dict:
            node = cfg_dict[prefix]
            for p in parts[i:]:
                if not isinstance(node, dict):
                    return default
                node = node.get(p, None)
                if node is None:
                    return default
            return node

    return default



def _artifact_urls(run):
    """
    Fetch URLs for test_predictions.json and output.log in a single files() call.
    Only called when --artifact-urls is passed (adds ~1 API request per run).
    """
    targets = {"test_predictions.json", "output.log"}
    found = {}
    try:
        for f in run.files():
            name = f.name.split("/")[-1]
            if name in targets:
                found[name] = f.url
                if len(found) == len(targets):
                    break
    except Exception:
        pass
    return found


def _per_transform_detail(run_summary):
    """
    Extract per-transformation-suite scalars from the run summary.
    Keeps only numeric values — skips WandB Table objects logged under the same prefix.
    These go into the JSON only (too many columns for CSV).
    """
    aggregate_suffixes = ("_std", "_worst10_avg", "_best10_avg")
    detail = {}
    for key, value in run_summary.items():
        if not key.startswith("test/id_per_transform_"):
            continue
        if any(key.endswith(suf) for suf in aggregate_suffixes):
            continue
        if not isinstance(value, (int, float)):
            continue
        short_key = key[len("test/"):]
        detail[short_key] = _safe(value)
    return detail


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_run_info(run, fetch_artifact_urls: bool = False) -> dict:
    """Return a flat dict with all information we care about for one run."""
    summary = dict(run.summary)
    cfg = dict(run.config) if run.config else {}

    row = {}

    # -- Run metadata --
    row["run_id"]    = run.id
    row["run_name"]  = run.name
    row["run_state"] = run.state
    # run_url: clickable web link to the run page (plots, tables, predictions, checkpoints, ...)
    row["run_url"]   = run.url
    # run_path: API path for programmatic access via wandb.Api().run(run_path)
    path = run.path
    row["run_path"]  = "/".join(path) if isinstance(path, list) else str(path)

    # created_at: read from _attrs directly (run.created_at maps to the same value)
    row["created_at"] = run._attrs.get("createdAt")

    # -- Config fields --
    for cfg_key, col_name in CONFIG_FIELDS.items():
        row[col_name] = _safe(_get_nested(cfg, cfg_key))

    # -- Summary / metric fields --
    for key in SUMMARY_FIELDS:
        col_name = key.replace("/", "__").replace(".", "_")
        row[col_name] = _safe(summary.get(key, None))

    # -- Artifact links (opt-in via --artifact-urls, adds ~1 API call per run) --
    if fetch_artifact_urls:
        artifacts = _artifact_urls(run)
        row["artifact_predictions_url"] = artifacts.get("test_predictions.json")
        row["artifact_output_log_url"]  = artifacts.get("output.log")
    else:
        row["artifact_predictions_url"] = None
        row["artifact_output_log_url"]  = None

    return row


def extract_run_info_full(run, fetch_artifact_urls: bool = False) -> dict:
    """
    Like extract_run_info but also includes per-transform detail and raw config
    for the JSON output.
    """
    flat = extract_run_info(run, fetch_artifact_urls=fetch_artifact_urls)
    summary = dict(run.summary)

    full = dict(flat)
    full["per_transform_detail"] = _per_transform_detail(summary)
    full["raw_config"] = {k: _safe(v) for k, v in (run.config or {}).items()}
    return full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Export WandB run results to CSV + JSON.")
    p.add_argument("--entity",        default="VisReas-ETHZ",        help="WandB entity (default: VisReas-ETHZ)")
    p.add_argument("--project",       required=True,                 help="WandB project name (e.g. compgen-reasoning)")
    p.add_argument("--output-dir",    default=".",                  help="Directory for output files")
    p.add_argument("--output-prefix", default="wandb_results",      help="Filename prefix")
    p.add_argument("--state",         default=None,
                   choices=["finished", "running", "crashed", "failed"],
                   help="Filter runs by state (default: all states)")
    p.add_argument("--runs", nargs="*", default=None,
                   help="Specific run IDs to export (default: all runs)")
    p.add_argument("--user", default=None,
                   help="Only include runs by this WandB username (e.g. ktroyan)")
    p.add_argument("--exclude-user", default=None,
                   help="Exclude runs by this WandB username (e.g. Yassine Taoudi-Benchekroun)")
    p.add_argument("--artifact-urls", action="store_true",
                   help="Fetch URLs for test_predictions.json and output.log (adds ~1 API call per run, slow for large projects)")
    p.add_argument("--sort-by", nargs="+", default=["created_at"],
                   metavar="COL",
                   help="Column(s) to sort rows by (default: created_at). Prefix with '-' for descending, e.g. --sort-by -test__id_grid_acc model_name")
    p.add_argument("--cols", nargs="+", default=None,
                   metavar="COL",
                   help="Explicit column order for the CSV. Unlisted columns are appended at the end.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    project_path = f"{args.entity}/{args.project}"

    print(f"Fetching runs from {project_path} ...")

    filters = {"state": args.state} if args.state else None

    if args.runs:
        all_runs = [api.run(f"{project_path}/{rid}") for rid in args.runs]
    else:
        all_runs = list(api.runs(project_path, filters=filters))

    # Show who created these runs (helpful for --user / --exclude-user)
    usernames = sorted({r._attrs.get("user", {}).get("username", "unknown") for r in all_runs})
    print(f"Found {len(all_runs)} run(s). Creators: {usernames}")

    if args.user or args.exclude_user:
        def _creator(r):
            # run._attrs["user"]["username"] is the individual WandB username of whoever
            # created the run. run.username / run.entity both return the team entity instead.
            try:
                return r._attrs["user"]["username"]
            except (KeyError, TypeError, AttributeError):
                return None

        before = len(all_runs)
        all_runs = [
            r for r in all_runs
            if (args.user is None or _creator(r) == args.user)
            and (args.exclude_user is None or _creator(r) != args.exclude_user)
        ]
        print(f"  After user filter: {len(all_runs)} run(s) (dropped {before - len(all_runs)})")

    if args.artifact_urls:
        print("  (--artifact-urls enabled: fetching file URLs per run, this may be slow)")

    flat_rows = []
    full_records = []

    for i, run in enumerate(all_runs, 1):
        print(f"  [{i:>4}/{len(all_runs)}] {run.id}  {run.name}  ({run.state})")
        try:
            flat_rows.append(extract_run_info(run, fetch_artifact_urls=args.artifact_urls))
            full_records.append(extract_run_info_full(run, fetch_artifact_urls=args.artifact_urls))
        except Exception as e:
            print(f"    WARNING: failed to extract run {run.id}: {e}")

    # -- CSV --
    df = pd.DataFrame(flat_rows)

    # -- Column order --
    if args.cols:
        # User-specified order; any unlisted columns are appended at the end
        leading = [c for c in args.cols if c in df.columns]
        unknown = [c for c in args.cols if c not in df.columns]
        if unknown:
            print(f"  WARNING: unknown --cols columns (ignored): {unknown}")
        trailing = [c for c in df.columns if c not in leading]
        df = df[leading + trailing]
    else:
        # Default order: metadata | config | metrics | artifacts
        meta_cols     = [c for c in df.columns if c.startswith("run_") or c == "created_at"]
        config_cols   = [c for c in CONFIG_FIELDS.values() if c in df.columns]
        artifact_cols = [c for c in df.columns if c.startswith("artifact_")]
        metric_cols   = [c for c in df.columns if c not in meta_cols and c not in config_cols and c not in artifact_cols]
        df = df[meta_cols + config_cols + metric_cols + artifact_cols]

    # -- Row order --
    sort_cols = []
    sort_asc  = []
    for col in args.sort_by:
        descending = col.startswith("-")
        col_name   = col.lstrip("-")
        if col_name not in df.columns:
            print(f"  WARNING: unknown --sort-by column '{col_name}' (ignored)")
            continue
        sort_cols.append(col_name)
        sort_asc.append(not descending)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=sort_asc, na_position="last")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(args.output_dir, f"{args.output_prefix}_{ts}.csv")
    json_path = os.path.join(args.output_dir, f"{args.output_prefix}_{ts}.json")

    df.to_csv(csv_path, index=False)
    print(f"\nCSV  -> {csv_path}  ({len(df)} rows, {len(df.columns)} columns)")

    with open(json_path, "w") as f:
        json.dump(full_records, f, indent=2, default=str)
    print(f"JSON -> {json_path}  ({len(full_records)} records)")

    # -- Field coverage summary --
    print("\n--- Metric coverage across all runs ---")
    numeric_cols = df.select_dtypes(include="number").columns
    coverage = (df[numeric_cols].notna().mean() * 100).sort_values(ascending=False)
    for col, pct in coverage.items():
        print(f"  {col:<60}  {pct:5.1f}%")


if __name__ == "__main__":
    main()
