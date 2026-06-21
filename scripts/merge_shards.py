"""
Merge per-shard MS3 evaluation outputs into a single combined deliverable.

Each shard ran `run_evaluation(..., shard_index=K, shard_total=N)` on its own
machine, producing a directory like:

    shard_0/
      checkpoint.json
      predictions_ms3.json
      evaluation_metrics_ms3.{csv,json}
      runtraces/runtrace_<id>.json   (one file per contract in this shard)
      runtraces_ms3.zip

This script reads all shard directories, concatenates per-contract data, and
emits a single combined output directory with:

    runtraces/runtrace_<id>.json     — every contract from every shard
    predictions_ms3.json             — verdicts from every contract
    evaluation_metrics_ms3.{csv,json} — re-aggregated from combined per-verdict scores
    evaluation_metrics_combined.csv  — MS1 + combined MS3 row (§5b deliverable)
    runtraces_ms3.zip                — zip of the merged runtraces (§5c deliverable)

Usage
-----
    python scripts/merge_shards.py \
        --shard-dir results/ms3/shard_0 \
        --shard-dir results/ms3/shard_1 \
        --shard-dir results/ms3/shard_2 \
        --shard-dir results/ms3/shard_3 \
        --shard-dir results/ms3/shard_4 \
        --output-dir results/ms3/merged \
        --ms1-csv results/evaluation_metrics.csv

Or — if each friend zips their output directory and you've extracted them
into `results/ms3/shards/`:

    python scripts/merge_shards.py \
        --shards-parent results/ms3/shards \
        --output-dir results/ms3/merged
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.evaluate_ms3 import (  # noqa: E402
    aggregate_metrics,
    write_csv,
    write_combined_csv,
)


def _load_shard(shard_dir: Path) -> Dict[str, Any]:
    """Read everything we need from one shard directory."""
    checkpoint = shard_dir / "checkpoint.json"
    runtraces_dir = shard_dir / "runtraces"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{shard_dir}: missing checkpoint.json — was the shard actually run?"
        )

    cp = json.loads(checkpoint.read_text(encoding="utf-8"))
    return {
        "shard_dir":          shard_dir,
        "done":               cp.get("done", {}),
        "per_verdict_scores": cp.get("per_verdict_scores", []),
        "latencies":          cp.get("latencies", []),
        "skipped":            cp.get("skipped", []),
        "runtraces_dir":      runtraces_dir if runtraces_dir.exists() else None,
    }


def merge_shards(
    shard_dirs: List[Path],
    output_dir: Path,
    ms1_csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Combine N shard directories into a single output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_runtraces = output_dir / "runtraces"
    out_runtraces.mkdir(exist_ok=True)

    shards = [_load_shard(p) for p in shard_dirs]

    print(f"[merge] loaded {len(shards)} shard(s)")

    # ── Combine per-shard state ──────────────────────────────────────────────
    all_done: Dict[str, Dict[str, Any]] = {}
    all_scores: List[Dict[str, Any]] = []
    all_latencies: List[float] = []
    all_skipped: List[str] = []
    duplicate_contract_ids: List[str] = []

    for shard in shards:
        for c_id, payload in shard["done"].items():
            if c_id in all_done:
                duplicate_contract_ids.append(c_id)
                continue  # first writer wins
            all_done[c_id] = payload

        all_scores.extend(shard["per_verdict_scores"])
        all_latencies.extend(shard["latencies"])
        all_skipped.extend(shard["skipped"])

    if duplicate_contract_ids:
        print(
            f"[merge] WARNING: {len(duplicate_contract_ids)} contract(s) appeared in "
            f"more than one shard (kept first occurrence): "
            f"{duplicate_contract_ids[:6]}{'...' if len(duplicate_contract_ids) > 6 else ''}",
            file=sys.stderr,
        )

    # ── Copy runtraces from every shard into the combined directory ──────────
    runtrace_copied = 0
    for shard in shards:
        rd = shard["runtraces_dir"]
        if rd is None:
            continue
        for src in rd.glob("runtrace_*.json"):
            dest = out_runtraces / src.name
            if dest.exists():
                continue  # keep the first
            shutil.copy2(src, dest)
            runtrace_copied += 1
    print(f"[merge] copied {runtrace_copied} runtrace files")

    # ── Write combined predictions in stable contract_id order ───────────────
    sorted_ids = sorted(all_done.keys(), key=lambda s: (len(s), s))
    all_predictions = [
        {
            "contract_id": c_id,
            "latency_ms":  all_done[c_id]["latency_ms"],
            "verdicts":    all_done[c_id]["verdicts"],
        }
        for c_id in sorted_ids
    ]
    (output_dir / "predictions_ms3.json").write_text(
        json.dumps(all_predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Re-aggregate metrics from the union of per-verdict scores ────────────
    metrics = aggregate_metrics(all_scores, all_latencies)
    metrics["skipped"]             = sorted(set(all_skipped))
    metrics["shard_count"]         = len(shards)
    metrics["duplicate_contracts"] = duplicate_contract_ids
    metrics["merged_at"]           = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"

    (output_dir / "evaluation_metrics_ms3.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(metrics, output_dir / "evaluation_metrics_ms3.csv")

    # ── §5b: combined CSV (MS1 + merged MS3) ─────────────────────────────────
    write_combined_csv(
        ms3_metrics=metrics,
        ms1_csv_path=ms1_csv_path,
        out_path=output_dir / "evaluation_metrics_combined.csv",
    )

    # ── §5c: zip every runtrace ──────────────────────────────────────────────
    zip_base = output_dir / "runtraces_ms3"
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=out_runtraces)
    metrics["runtrace_archive"] = Path(archive).name

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="merge_shards",
        description="Combine per-shard MS3 evaluation outputs into a single deliverable.",
    )
    parser.add_argument(
        "--shard-dir", action="append", type=Path, default=[],
        help="Path to one shard's output directory (repeat for each shard).",
    )
    parser.add_argument(
        "--shards-parent", type=Path, default=None,
        help="Convenience: parent directory whose immediate subdirectories each "
             "contain a checkpoint.json (one per shard).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/ms3/merged"),
        help="Where to write the merged outputs.",
    )
    parser.add_argument(
        "--ms1-csv", type=Path,
        default=Path("results/evaluation_metrics.csv"),
        help="MS1 metrics CSV for the §5b combined CSV (skipped if missing).",
    )
    args = parser.parse_args()

    shard_dirs: List[Path] = list(args.shard_dir)
    if args.shards_parent:
        for sub in sorted(args.shards_parent.iterdir()):
            if sub.is_dir() and (sub / "checkpoint.json").exists():
                shard_dirs.append(sub)

    if not shard_dirs:
        print("error: provide --shard-dir (repeatable) or --shards-parent", file=sys.stderr)
        return 2

    ms1_csv = args.ms1_csv if args.ms1_csv and args.ms1_csv.exists() else None

    print(f"[merge] inputs: {len(shard_dirs)} shard(s)")
    for d in shard_dirs:
        print(f"          - {d}")
    print(f"[merge] output: {args.output_dir}")
    print(f"[merge] MS1 CSV: {ms1_csv or '(none — combined CSV will only have MS3 row)'}")

    metrics = merge_shards(
        shard_dirs   = shard_dirs,
        output_dir   = args.output_dir,
        ms1_csv_path = ms1_csv,
    )

    print("\n=== Merged aggregate metrics ===")
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "per_hypothesis"},
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
