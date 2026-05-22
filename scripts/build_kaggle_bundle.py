"""
Build a single self-contained Python file that runs the MS3 evaluation on Kaggle.

Concatenates every source module the agentic pipeline needs into one giant
.py file, inlines `playbook.yaml` as a string constant, and appends a
`__main__` block that:

  1. Pulls Kaggle secrets into the env
  2. Loads a BASE Qwen2.5-7B-Instruct on the T4 GPU via Unsloth (4-bit)
  3. Installs the LocalInferenceClient shim so every agent uses the local model
  4. Builds the orchestrator + retriever
  5. Runs scripts.evaluate_ms3.run_evaluation over the test split
  6. Writes the deliverables (predictions, runtraces, combined CSV)

Re-run this script whenever team members merge new code so the bundle stays
fresh. The bundle is committed to the repo so users can grab it directly
without running the bundler.

Usage
-----
    python scripts/build_kaggle_bundle.py

Output
------
    kaggle_ms3_eval.py   (at repo root)
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "kaggle_ms3_eval.py"

# Source files in dependency order — the bundler concatenates them as listed.
BUNDLE_FILES: List[str] = [
    "src/retrieval/base.py",
    "src/retrieval/vector_rag.py",
    "scripts/graphrag_utils.py",        # peer module the retriever sys.path-hacks
    "src/retrieval/graphrag_retriever.py",
    "src/utils/contract_loader.py",
    "src/utils/runtrace.py",
    "src/enrichment/playbook_enricher.py",
    "src/agent/history.py",
    "src/agent/intent_router.py",
    "src/agent/conversation_agent.py",
    "src/agent/hypothesis_analyst.py",
    "src/agent/reviewer_agent.py",
    "src/agent/hypothesis_pipeline.py",
    "src/agent/orchestrator.py",
    "src/agent/local_inference_client.py",
    "scripts/evaluate_ms3.py",
]

# Regexes for lines that need to be removed/commented from each source file
# because the symbols they reference are already defined elsewhere in the bundle.
STRIP_PATTERNS = [
    re.compile(r"^\s*from\s+\.\.?[\w\.]*\s+import\s+"),    # `from ..foo import` / `from .foo import`
    re.compile(r"^\s*from\s+src\.[\w\.]+\s+import\s+"),    # `from src.foo import`
    re.compile(r"^\s*import\s+src\.[\w\.]+"),              # `import src.foo`
    re.compile(r"^\s*from\s+__future__\s+import\s+annotations"),  # only need one at the top
    re.compile(r"^\s*from\s+graphrag_utils\s+import\s+"),  # graphrag_utils.py is bundled inline
    re.compile(r"^\s*import\s+graphrag_utils"),
    # NOTE: the `_SCRIPTS_DIR = ...`, `if _SCRIPTS_DIR not in sys.path:`, and
    # `sys.path.insert(0, _SCRIPTS_DIR)` lines in src/retrieval/graphrag_retriever.py
    # are intentionally LEFT IN the bundle. Stripping the inner sys.path.insert
    # line while keeping the `if` indented becomes a syntax error; letting all
    # three lines stay is a harmless no-op (it adds a non-existent path to
    # sys.path, which Python silently tolerates).
]


_MAIN_BLOCK_RE = re.compile(r'^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:')


def _process_module(rel_path: str) -> str:
    """
    Read a source file and strip imports that don't apply in a flat bundle.
    Replaces stripped lines with `pass` instead of a comment so blocks like
    `if cond:\n    from src.x import Y` don't become empty after stripping.

    Also strips top-level `if __name__ == "__main__":` blocks: each bundled
    module may have its own CLI entrypoint, but only the bundle's footer
    `__main__` should actually run. Without this, the first __main__ block
    encountered exits before the Kaggle entrypoint (which writes the inlined
    playbook) ever runs.

    Also handles multi-line imports: if a stripped line opens an unmatched
    parenthesis, continuation lines are consumed until the closing `)`.
    """
    src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    out_lines: List[str] = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _MAIN_BLOCK_RE.match(line):
            # Consume this line plus all indented/blank lines underneath.
            i += 1
            while i < len(lines) and (lines[i].strip() == "" or lines[i][:1] in (" ", "\t")):
                i += 1
            continue
        if any(p.match(line) for p in STRIP_PATTERNS):
            indent = re.match(r"^(\s*)", line).group(1)
            out_lines.append(f"{indent}pass  # [bundled] {line.lstrip()}")
            # If this stripped line opens an unbalanced "(", consume continuation
            # lines until the matching ")" so we don't orphan import targets.
            open_count = line.count("(") - line.count(")")
            j = i
            while open_count > 0 and j + 1 < len(lines):
                j += 1
                open_count += lines[j].count("(") - lines[j].count(")")
                # Drop the continuation entirely (don't emit it)
            i = j + 1
        else:
            out_lines.append(line)
            i += 1
    return "\n".join(out_lines)


def _read_playbook() -> str:
    return (REPO_ROOT / "playbook.yaml").read_text(encoding="utf-8")


# ────────────────────────────────────────────────────────────────────────────
# Bundle template chunks
# ────────────────────────────────────────────────────────────────────────────

HEADER_TEMPLATE = '''#!/usr/bin/env python3
"""
BetterCallNLI MS3 — single-file Kaggle evaluation bundle
========================================================

Auto-generated by scripts/build_kaggle_bundle.py on {generated_at}.
Do NOT edit by hand — re-run the bundler when team members merge changes.

Bundled source files (in order):
{file_list}

Spec compliance (MS3 §2f)
-------------------------
This bundle loads the BASE Qwen2.5-7B-Instruct (Qwen2.5 family, no LoRA
adapter). Do NOT change --model to point at any fine-tuned adapter.

Quick start on Kaggle
---------------------
1. New Notebook → Settings → Accelerator → GPU T4 x2
2. Add-ons → Secrets:
     CHROMA_API_KEY                 (for --retrieval vector)
     NEO4J_URI / _USERNAME / _PASSWORD  (for --retrieval graphrag)
3. Upload this single file to the notebook (or attach as Dataset)
4. Run a cell:
     !pip install -q --upgrade unsloth unsloth_zoo
     !pip install -q -U 'bitsandbytes>=0.46.1'
     !pip install -q rich tqdm pandas pyyaml sentence-transformers \\
                     chromadb neo4j huggingface_hub kagglehub
     !python kaggle_ms3_eval.py --retrieval vector --output-dir /kaggle/working/outputs/ms3
5. Download /kaggle/working/outputs/ms3/runtraces_ms3.zip (§5c) and
   evaluation_metrics_combined.csv (§5b).
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════
# Top-of-bundle constants
# ════════════════════════════════════════════════════════════════════════════

# Task 4 modules are always present in this bundle (PlaybookEnricher and
# RuntraceFormatter are defined inline below), so the soft-import flags in
# the bundled scripts/evaluate_ms3.py block don't need to do their dance.
_HAS_TASK4_ENRICHER  = True
_HAS_TASK4_FORMATTER = True

# Inlined playbook.yaml — written to disk at runtime so PlaybookEnricher and
# Orchestrator can read it via their normal file-path API.
PLAYBOOK_YAML = r"""{playbook_yaml}"""

'''


FOOTER_BLOCK = '''

# ════════════════════════════════════════════════════════════════════════════
# Main entrypoint — `python kaggle_ms3_eval.py [flags]`
# ════════════════════════════════════════════════════════════════════════════

def _kaggle_setup_env() -> None:
    """Pull Kaggle Secrets into os.environ when running inside a Kaggle kernel."""
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        for k in ("HF_TOKEN", "CHROMA_API_KEY",
                  "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
            try:
                import os
                os.environ[k] = secrets.get_secret(k)
            except Exception:
                pass
    except ImportError:
        pass
    import os
    # The Orchestrator factory checks for HF_TOKEN but the local shim ignores it.
    os.environ.setdefault("HF_TOKEN", "local-model-stub")


def _kaggle_main(args) -> int:
    import os, json
    from pathlib import Path

    # 1. Write the inlined playbook to disk so PlaybookEnricher can read it
    playbook_path = Path(args.output_dir).parent / "playbook.yaml"
    playbook_path.parent.mkdir(parents=True, exist_ok=True)
    playbook_path.write_text(PLAYBOOK_YAML, encoding="utf-8")
    print(f"[bundle] wrote playbook to {playbook_path}")

    # 2. Pull Kaggle Secrets into env
    _kaggle_setup_env()

    # 3. Load BASE model (no LoRA — §2f) via Unsloth
    print(f"[bundle] loading {args.model} in 4-bit on GPU 0...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = args.model,
        max_seq_length  = args.max_seq_len,
        load_in_4bit    = True,
        device_map      = {"": 0},
    )
    FastLanguageModel.for_inference(model)
    print("[bundle] model ready")

    # 4. Install the LocalInferenceClient shim so every agent uses the GPU model
    install_as_global_client(model, tokenizer, device="cuda")

    # 5. Build retriever directly (skip build_orchestrator factory: it reads .env)
    if args.retrieval == "vector":
        retriever = VectorRAGRetriever()
        if not retriever.is_ready():
            raise RuntimeError("VectorRAGRetriever not ready — check CHROMA_API_KEY secret")
    else:
        retriever = GraphRAGRetriever(
            uri      = os.environ["NEO4J_URI"],
            username = os.environ["NEO4J_USERNAME"],
            password = os.environ["NEO4J_PASSWORD"],
        )
        if not retriever.connect():
            raise RuntimeError("GraphRAGRetriever connect failed — check NEO4J_* secrets")

    orchestrator = Orchestrator(
        retriever     = retriever,
        hf_token      = os.environ["HF_TOKEN"],
        playbook_path = str(playbook_path),
        model         = args.model,
    )
    print(f"[bundle] orchestrator built (retrieval={orchestrator.retriever.mode})")

    # 6. Load contracts
    contracts = get_test_contracts()
    print(f"[bundle] loaded {len(contracts)} test contracts")

    # 7. Run evaluation
    ms1_csv = Path(args.ms1_csv) if args.ms1_csv and Path(args.ms1_csv).exists() else None
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=args.limit or len(contracts), desc="evaluate")
        def _cb(i, n, c_id, *, status="ok", latency_ms=0.0):
            bar.set_postfix_str(f"{c_id} [{status}] {latency_ms/1000:.1f}s")
            bar.update(1)
    except ImportError:
        bar = None
        def _cb(i, n, c_id, *, status="ok", latency_ms=0.0):
            print(f"  [{i}/{n}] {c_id} [{status}] {latency_ms/1000:.1f}s")

    metrics = run_evaluation(
        orchestrator  = orchestrator,
        contracts     = contracts,
        output_dir    = Path(args.output_dir),
        limit         = args.limit,
        progress_cb   = _cb,
        playbook_path = playbook_path,
        ms1_csv_path  = ms1_csv,
        shard_index   = args.shard_index,
        shard_total   = args.shard_total,
    )
    if bar:
        bar.close()

    print()
    print("=== Aggregate metrics ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_hypothesis"}, indent=2))
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="kaggle_ms3_eval")
    parser.add_argument("--retrieval",  choices=["vector", "graphrag"], default="vector")
    parser.add_argument("--output-dir", default="/kaggle/working/outputs/ms3")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Smoke-test cap (e.g. --limit 5)")
    parser.add_argument("--model",      default="unsloth/Qwen3-4B-bnb-4bit",
                        help="BASE model only — do not point at a fine-tuned adapter (§2f)")
    parser.add_argument("--max-seq-len", type=int, default=8192,
                        help="Contract + retrieved precedents + system prompt routinely 5-7k tokens; "
                             "2048 silently truncates the contract. Bump to 16384 for very long NDAs.")
    parser.add_argument("--ms1-csv",    default=None,
                        help="Path to existing MS1 evaluation_metrics.csv for the combined CSV (§5b)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="0-indexed shard for parallel runs across machines (default 0)")
    parser.add_argument("--shard-total", type=int, default=1,
                        help="Total number of shards; merge with scripts/merge_shards.py afterwards")
    args = parser.parse_args()
    raise SystemExit(_kaggle_main(args))
'''


# ────────────────────────────────────────────────────────────────────────────
# Bundler
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    missing = [f for f in BUNDLE_FILES if not (REPO_ROOT / f).exists()]
    if missing:
        print(f"ERROR: missing source files: {missing}", file=sys.stderr)
        return 1

    parts: List[str] = []

    # Header + inlined playbook
    parts.append(HEADER_TEMPLATE.format(
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        file_list    = "\n".join(f"  - {f}" for f in BUNDLE_FILES),
        playbook_yaml = _read_playbook(),
    ))

    # Each module's content with a section banner
    for rel in BUNDLE_FILES:
        parts.append("\n# " + "=" * 76)
        parts.append(f"# {rel}")
        parts.append("# " + "=" * 76)
        parts.append(_process_module(rel))

    # Main block
    parts.append(FOOTER_BLOCK)

    OUTPUT_FILE.write_text("\n".join(parts), encoding="utf-8")
    print(f"[bundler] wrote {OUTPUT_FILE.name} "
          f"({OUTPUT_FILE.stat().st_size:,} bytes, "
          f"{OUTPUT_FILE.read_text(encoding='utf-8').count(chr(10)):,} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
