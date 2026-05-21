# Milestone 3 — Run Instructions (Task 5: CLI + Evaluation)

This document covers how to run the MS3 entry points and how they map to the [MS3 spec](./MS3_spec.pdf).

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
| §2a — 17 Hypotheses Analysis mode (JSON output) | `cli.py --mode analyze [--output result.json]` |
| §2b — Conversation Mode with citations | `cli.py --mode converse` |
| §2c — Auditability runtraces (per contract + per conversation session) | `scripts/evaluate_ms3.py:write_runtrace` (per contract); `cli.py:_write_conversation_runtrace` (per session) |
| §2d — Apply playbook from MS1 (no edits) | `scripts/evaluate_ms3.py:apply_playbook` reads the unmodified `playbook.yaml` |
| §2e — CLI with flags + multi-turn conversation | `cli.py` |
| §2f — No fine-tuned model | All agents use `Qwen/Qwen2.5-7B-Instruct` on HF Serverless |
| §2g — Vector RAG **and** GraphRAG | `--retrieval vector` / `--retrieval graphrag` |
| §2h — Runtraces include tool_calls with `{name, args, output, count}` per agent | `scripts/evaluate_ms3.py:normalize_tool_calls` |
| §3 — Evaluation on MS1 test split with the same metrics | `cli.py --mode evaluate` → `predictions_ms3.json`, runtraces, `evaluation_metrics_ms3.csv` |
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
python cli.py --mode converse --contract path/to/contract.txt --retrieval graphrag
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
python cli.py --mode analyze --contract path/to/contract.txt --retrieval graphrag --show-cards --output result.json
```

- Runs the full hypothesis pipeline once and prints a summary table.
- `--show-cards` adds a labelled panel per verdict.
- `--output result.json` writes the result dict (verdicts + tool_calls) to disk.
- Verdicts are enriched with deterministic playbook fields (§3c): `status`, `severity`, `action`, `criticality`, `rationale`.

### 2c. Evaluate mode (§3, batch over the MS1 test split)

```powershell
python cli.py --mode evaluate --retrieval graphrag --output-dir results/ms3
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
| `--ms1-csv results/evaluation_metrics.csv` | MS1 CSV to merge into the combined CSV (default points to the existing MS1 file in this repo) |
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

The same evaluation can be launched without going through `cli.py`:

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

### How the shim works

`install_as_global_client(model, tokenizer)`:
1. Replaces `huggingface_hub.InferenceClient` with a `LocalInferenceClient` subclass that captures the loaded model+tokenizer.
2. Walks the already-imported agent modules (`src.agent.intent_router`, `src.agent.conversation_agent`, `src.agent.hypothesis_analyst`, `src.agent.reviewer_agent`) and rebinds their local `InferenceClient` symbol — necessary because they did `from huggingface_hub import InferenceClient` at module load.

After step 2, every agent that calls `self._client.chat_completion(...)` runs against the GPU model. No agent code changes; the agents don't know they're not talking to HF Serverless.

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
