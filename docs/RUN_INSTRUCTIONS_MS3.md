# Milestone 3 — Run Instructions (Task 5: CLI + Evaluation)

This document covers how to run the MS3 entry points and how they map to the [MS3 spec](./spec/m3.pdf).

For MS1 training & the MS1 inference notebook see [`RUN_INSTRUCTIONS.md`](./RUN_INSTRUCTIONS.md).

---

## Important: do NOT use the fine-tuned MS1 model

Per **MS3 spec §2f**:

> Do not use the fine-tuned model in your implementation, you can use any one or more of the models from the same family of the fine-tuned model.

The agentic system is the comparison baseline against the MS1 fine-tune, so every LLM call in the pipeline must hit a *base* model from the same family. The existing agents in this repo all call **`Qwen/Qwen2.5-7B-Instruct`** via the HuggingFace Serverless Inference API (`huggingface_hub.InferenceClient`). No local model loading; no LoRA adapter at runtime.

---

## Spec coverage at a glance

| Spec | Where it's implemented |
|---|---|
| §2a — 17 Hypotheses Analysis mode (JSON output) | `apps/cli.py --mode analyze [--output result.json]` |
| §2b — Conversation Mode with citations | `apps/cli.py --mode converse` |
| §2c — Auditability runtraces (per contract + per conversation session) | `scripts/evaluate_ms3.py:write_runtrace` (per contract); `apps/cli.py:_write_conversation_runtrace` (per session) |
| §2d — Apply playbook from MS1 (no edits) | `scripts/evaluate_ms3.py:apply_playbook` reads the unmodified `playbook.yaml` |
| §2e — CLI with flags + multi-turn conversation | `apps/cli.py` |
| §2f — No fine-tuned model | All agents use `Qwen/Qwen2.5-7B-Instruct` on HF Serverless |
| §2g — Vector RAG **and** GraphRAG | `--retrieval vector` / `--retrieval graphrag` |
| §2h — Runtraces include tool_calls with `{name, args, output, count}` per agent | `scripts/evaluate_ms3.py:normalize_tool_calls` |
| §3 — Evaluation on MS1 test split with the same metrics | `apps/cli.py --mode evaluate` → `predictions_ms3.json`, runtraces, `evaluation_metrics_ms3.csv` |
| §5b — ONE final CSV with MS1 + MS3 metrics | `evaluation_metrics_combined.csv` |
| §5c — Zip of all runtraces | `runtraces_ms3.zip` (auto-emitted by the evaluation runner) |

---

## 1. Local install

```powershell
# from the repo root
pip install -r requirements_ms3.txt
```

Required `.env` keys (place in repo root):

```
HF_TOKEN=hf_...                    # HuggingFace Serverless API (required, all modes)
NEO4J_URI=neo4j+s://...            # required for --retrieval graphrag
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
CHROMA_API_KEY=ck-...              # required for --retrieval vector
```

---

## 2. CLI usage

The CLI is the spec-mandated entry point (§2e). Pick a mode via `--mode`:

### 2a. Conversation mode (§2b, multi-turn legal-assistant chat)

```powershell
python apps/cli.py --mode converse --contract path/to/contract.txt --retrieval graphrag
```

- Loads one contract, then prompts for questions interactively.
- Each user message goes through the **IntentRouter**, which dispatches to either the `ConversationAgent` (free-form Q&A with verbatim citations) or the `HypothesisPipeline` (full 17-hypothesis review).
- Type `exit`, `quit`, or `Ctrl+C` to end the session. Type `reset` to clear history without exiting.
- On exit a per-session runtrace is written (§2c — see `--session-runtrace`).

| Flag | Purpose |
|---|---|
| `--save-history history.json` | Persist `ConversationHistory` on every turn |
| `--session-runtrace path.json` | Override location of the per-session runtrace (defaults to `results/ms3/conversation_runtraces/session_<id>.json`) |
| `-v`, `--verbose` | Print the `tool_calls` trace under each response |

### 2b. Analyze mode (§2a, one-shot 17-hypothesis review)

```powershell
python apps/cli.py --mode analyze --contract path/to/contract.txt --retrieval graphrag --show-cards --output result.json
```

- Runs the full hypothesis pipeline once and prints a summary table.
- `--show-cards` adds a labelled panel per verdict.
- `--output result.json` writes the result dict (verdicts + tool_calls) to disk.
- Verdicts are enriched with deterministic playbook fields (§3c): `status`, `severity`, `action`, `criticality`, `rationale`.

### 2c. Evaluate mode (§3, batch over the MS1 test split)

```powershell
python apps/cli.py --mode evaluate --retrieval graphrag --output-dir results/ms3
```

What it does:

1. Loads the **same** ContractNLI test split MS1 used (`get_test_contracts(...)`).
2. Runs the hypothesis pipeline on every contract.
3. Enriches verdicts with playbook policy fields (§3c).
4. Writes one runtrace per contract (§2c) with normalized tool_calls (§2h).
5. Aggregates the **same metrics MS1 used** (§3e):
   - `label_accuracy`, `groundedness_rate`, `quote_integrity_rate`, `avg_latency_ms`
6. Writes a **single combined CSV** with both MS1 and MS3 rows (§5b deliverable).
7. Zips every runtrace to `runtraces_ms3.zip` (§5c deliverable).

Useful flags:

| Flag | Purpose |
|---|---|
| `--limit 5` | Quick smoke run on the first N contracts |
| `--data-dir /path/to/contractnli/` | Skip the kagglehub download, use a local dataset copy |
| `--ms1-csv results/ms1/evaluation_metrics.csv` | MS1 CSV to merge into the combined CSV (default points to the existing MS1 file in this repo) |
| `--playbook playbook.yaml` | Override playbook location (default: repo-root playbook) |

Outputs (under `--output-dir`):

| File | Spec | Contents |
|---|---|---|
| `predictions_ms3.json` | §3a, §3b | All verdicts with labels + evidence per contract |
| `runtraces/runtrace_<id>.json` | §2c, §2h, §3d | Per-contract runtrace: verdicts + agent_traces + tool_calls (count normalized) |
| `evaluation_metrics_ms3.csv` | §3e | MS3 row, header identical to MS1 CSV |
| `evaluation_metrics_combined.csv` | §5b | MS1 row + MS3 row (deliverable) |
| `evaluation_metrics_ms3.json` | — | Per-hypothesis accuracy + confusion counts |
| `runtraces_ms3.zip` | §5c | All runtraces zipped (deliverable) |

---

## 3. Equivalence: standalone evaluation script

The same evaluation can be launched without going through `apps/cli.py`:

```powershell
python -m scripts.evaluate_ms3 --retrieval graphrag --output-dir results/ms3
```

Use this when scripting from CI or a notebook; the CLI is just a styled wrapper around it.

---

## 4. Final deliverables checklist (§5)

After running `--mode evaluate` end-to-end you should have:

- [x] **§5a Codebase on a new branch** — `feat/cli-evaluation` (off `ms-2`)
- [x] **§5b Final evaluation CSV with MS1 + MS3** — `results/ms3/evaluation_metrics_combined.csv`
- [x] **§5c Zip of runtraces** — `results/ms3/runtraces_ms3.zip`
- [ ] **§5d Contribution markdown** — `CONTRIBUTION.md` (team-wide doc, not part of Task 5)

---

## 5. Running on Kaggle (local base model, no HF Serverless)

For evaluation runs that hit the LLM ~6 000+ times (123 contracts × 17 hypotheses × up-to-3 attempts), HF Serverless quotas can be a bottleneck. The Kaggle path loads a **base** Qwen2.5 model directly on a T4 GPU and routes every agent's `chat_completion(...)` through it via a drop-in `LocalInferenceClient`. No HF API calls during the run.

**Spec compliance (§2f).** The model loaded is the **base** Qwen2.5-7B-Instruct — same family as your MS1 fine-tune, NO LoRA adapter attached. Do not point `MODEL_NAME` at your `qwen3-4B-nli-lora-adapter`.

### Steps

1. **Upload the repo as a Kaggle Dataset.** Zip the project root and add it to the notebook's *Add data* panel. Default mount path: `/kaggle/input/bettercallnli/BetterCallNLI` (update `REPO_DIR` in Cell 2 if different).
2. **Enable GPU.** *Settings → Accelerator → GPU T4 x2* (or any single T4 — the 4-bit 7B model fits in ~10 GB).
3. **Add Kaggle Secrets** under *Add-ons → Secrets*:
   - `CHROMA_API_KEY` (for `RETRIEVAL_MODE = "vector"`)
   - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` (for `RETRIEVAL_MODE = "graphrag"`)
   - `HF_TOKEN` is optional — only needed if Qwen weights require gated access. The shim ignores it for inference.
4. **Open** [`notebooks/run_ms3_kaggle.ipynb`](./notebooks/run_ms3_kaggle.ipynb) and run all cells. The flow:
   - Cell 1: install Unsloth + bitsandbytes + agent deps
   - Cell 2: configure paths, pull secrets, set `MODEL_NAME` (default `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`)
   - Cell 3: load the base model in 4-bit on GPU 0
   - Cell 4: `install_as_global_client(model, tokenizer)` — replaces `huggingface_hub.InferenceClient` with `LocalInferenceClient`
   - Cell 5: build orchestrator + 1-contract smoke
   - Cell 6: full evaluation (~2–4 hours on T4)
   - Cell 7: per-hypothesis breakdown + file listing
5. **Download** the artifacts via the *Output* tab:
   - `outputs/ms3/runtraces_ms3.zip` (§5c deliverable)
   - `outputs/ms3/evaluation_metrics_combined.csv` (§5b deliverable)

### Resuming after a Kaggle timeout

`run_evaluation()` checkpoints after **every contract** to `<output_dir>/checkpoint.json` (atomic write, safe to interrupt). The file holds the per-contract verdicts, accumulated per-verdict scores, latencies, and skipped list.

If your Kaggle session times out / restarts mid-run:

1. Re-run **Cells 1 → 5** to reload the model, install the shim, and rebuild the orchestrator.
2. Re-run **Cell 6** — `run_evaluation()` sees the existing `checkpoint.json`, prints
   ```
   [evaluate_ms3] resuming from checkpoint: 47 contracts already processed, 0 previously skipped
   ```
   …and continues from contract 48.

Per-contract runtraces (`runtraces/runtrace_<id>.json`) are also written incrementally as each contract finishes, so even without the checkpoint you have durable per-contract output on disk. The final aggregated `predictions_ms3.json`, `evaluation_metrics_*.{csv,json}`, and `runtraces_ms3.zip` are produced at the very end once every contract is done.

To start fresh instead of resuming, pass `resume=False` to `run_evaluation()` or delete `<output_dir>/checkpoint.json` before re-running Cell 6.

### Parallel runs across 5 machines (sharding)

The full evaluation makes ~6 000 LLM calls — ~3 hours on one T4. Split across **5 Kaggle accounts** it finishes in ~30 minutes.

**Setup:**

1. **Hand out one notebook per friend** from [`notebooks/shards/`](./notebooks/shards/):
   - `shard_0.ipynb` → friend 1 (processes contracts `[0:24]`)
   - `shard_1.ipynb` → friend 2 (processes contracts `[24:49]`)
   - `shard_2.ipynb` → friend 3 (processes contracts `[49:73]`)
   - `shard_3.ipynb` → friend 4 (processes contracts `[73:98]`)
   - `shard_4.ipynb` → you (processes contracts `[98:123]`)
2. **Each friend:**
   - Open their assigned notebook on Kaggle (GPU T4 x2 enabled)
   - Add `CHROMA_API_KEY` to Kaggle Secrets
   - Run All cells — the notebook downloads the latest bundle from GitHub, runs only its slice with checkpointing, then zips the output
   - Download `/kaggle/working/ms3_shard_<N>.zip` from the Output tab
   - Send the zip to whoever's doing the merge
3. **The merger:**
   - Extract all 5 zips into one parent directory:
     ```
     results/ms3/shards/
       shard_0/   ← extracted contents
       shard_1/
       shard_2/
       shard_3/
       shard_4/
     ```
   - Run:
     ```powershell
     python scripts/merge_shards.py \
         --shards-parent results/ms3/shards \
         --output-dir results/ms3/merged
     ```
   - Outputs in `results/ms3/merged/`:
     - `predictions_ms3.json` — every contract's verdicts
     - `runtraces/runtrace_<id>.json` — every per-contract runtrace
     - `evaluation_metrics_ms3.{csv,json}` — re-aggregated metrics from the union of all shards
     - `evaluation_metrics_combined.csv` — §5b deliverable (MS1 + merged MS3 row)
     - `runtraces_ms3.zip` — §5c deliverable

**Manual shard invocation** (CLI / bundle):
```bash
# bundled single-file path
!python kaggle_ms3_eval.py --retrieval vector \
    --shard-index 2 --shard-total 5 \
    --output-dir /kaggle/working/outputs/ms3_shard_2

# repo-installed CLI path
python apps/cli.py --mode evaluate --retrieval vector \
    --shard-index 2 --shard-total 5 \
    --output-dir results/ms3/shard_2
```

Each shard maintains its own `checkpoint.json`, so individual shards are independently resumable after Kaggle timeouts.

### How the shim works

`install_as_global_client(model, tokenizer)`:
1. Replaces `huggingface_hub.InferenceClient` with a `LocalInferenceClient` subclass that captures the loaded model+tokenizer.
2. Walks the already-imported agent modules (`src.agent.intent_router`, `src.agent.conversation_agent`, `src.agent.hypothesis_analyst`, `src.agent.reviewer_agent`) and rebinds their local `InferenceClient` symbol — necessary because they did `from huggingface_hub import InferenceClient` at module load.

After step 2, every agent that calls `self._client.chat_completion(...)` runs against the GPU model. No agent code changes; the agents don't know they're not talking to HF Serverless.

### Alternative: single-file bundle (`kaggle_ms3_eval.py`)

If you don't want to upload the whole repo as a Kaggle Dataset, [`kaggle_ms3_eval.py`](../notebooks/ms3/kaggle_ms3_eval.py) is a ~5000-line self-contained script that bundles every agent module, the playbook (inlined), Member 4's PlaybookEnricher and RuntraceFormatter, and the eval runner into one file. Drop it onto a Kaggle notebook and run:

```bash
!pip install -q --upgrade unsloth unsloth_zoo
!pip install -q -U 'bitsandbytes>=0.46.1'
!pip install -q rich tqdm pandas pyyaml sentence-transformers chromadb neo4j huggingface_hub kagglehub
!python kaggle_ms3_eval.py --retrieval vector --output-dir /kaggle/working/outputs/ms3
```

Same outputs as the notebook (`predictions_ms3.json`, `runtraces/`, `evaluation_metrics_combined.csv`, `runtraces_ms3.zip`). Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--retrieval`  | `vector`                            | `vector` (ChromaDB) or `graphrag` (Neo4j) |
| `--output-dir` | `/kaggle/working/outputs/ms3`       | Where to write deliverables |
| `--limit`      | none                                | Cap N contracts (smoke run) |
| `--model`      | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | BASE model only — do NOT point at a LoRA adapter (§2f) |
| `--max-seq-len`| `8192`                              | Bump to `16384` for very long NDA contracts; never drop to 2048 (silent truncation) |
| `--ms1-csv`    | none                                | MS1 metrics CSV to merge into the combined CSV (§5b) |

Regenerate the bundle when team members push changes by running:
```powershell
python scripts/build_kaggle_bundle.py
```
The bundler concatenates every module in dependency order, inlines `playbook.yaml`, strips relative imports, and emits the updated single file at `notebooks/ms3/kaggle_ms3_eval.py`.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `UnicodeEncodeError` on Windows | The CLI already calls `sys.stdout.reconfigure(encoding="utf-8")` at startup. If using an old PowerShell, set `$env:PYTHONIOENCODING="utf-8"`. |
| `EnvironmentError: HF_TOKEN is not set` | Add `HF_TOKEN=...` to your `.env`. |
| `GraphRAGRetriever failed to connect` | Check `NEO4J_*` in `.env` and that the Aura instance is running. |
| `VectorRAGRetriever could not connect` | Check `CHROMA_API_KEY` in `.env`. |
| `Playbook not found` | Pass `--playbook /full/path/to/playbook.yaml` or run from the repo root. |
| Combined CSV has no MS1 row | Either `--ms1-csv` doesn't point at the MS1 file, or the MS1 CSV is missing. |
