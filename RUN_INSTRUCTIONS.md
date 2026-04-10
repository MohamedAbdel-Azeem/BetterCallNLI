# Reproducible Run Instructions

## Prerequisites

- **Platform:** [Kaggle Notebooks](https://www.kaggle.com/) with **GPU T4 x2** accelerator enabled
- **Accounts:** Kaggle account, Hugging Face account with a [read-access token](https://huggingface.co/settings/tokens)
- **Time:** ~4.5 hours (training dominates)

---

## Step 1: Open the Notebook

1. Upload or open `notebooks/training_notebook.ipynb` in a Kaggle notebook session.
2. Under **Settings > Accelerator**, select **GPU T4 x2**.
3. Run all cells sequentially from top to bottom.

---

## Step 2: Install Dependencies

When the install cells run, the following packages are installed:

```bash
pip install --upgrade unsloth unsloth_zoo
pip install trl==1.0.0
pip install -U "bitsandbytes>=0.46.1"
```

When prompted, enter your Hugging Face API token:

```
Enter your HF token: <paste your token>
```

---

## Step 3: Download the Dataset

The dataset downloads automatically via `kagglehub` — no manual action needed:

```python
kagglehub.dataset_download('seiftarek158/contract-nli')
```

This fetches `train.json`, `dev.json`, and `test.json` (607 NDA documents, 17 hypotheses each).

---

## Step 4: Data Exploration & Preprocessing

Continue running cells sequentially. These cells:

1. Print label distributions, contract length stats, and evidence span analysis.
2. Flatten the dataset into `(document, hypothesis)` records.
3. Build training/inference prompts with truncation (skip first 10% of contract, cap at 1,400/1,700 tokens).
4. Serialize preprocessed splits to `/kaggle/working/preprocessed/`.

No manual configuration is needed — all parameters are set in the notebook.

---

## Step 5: Load Base Model & Configure QLoRA

Continue running cells. The notebook will:

1. Load `Qwen/Qwen3-4B-Instruct-2507` in 4-bit quantization via Unsloth.
2. Run a pre-training sanity check (single inference to show raw output).
3. Apply LoRA adapters (r=12, alpha=32, dropout=0.15) to attention layers.

---

## Step 6: Train

Continue running cells. Training starts automatically with:

- **Effective batch size:** 16 (2 per device x 8 gradient accumulation steps)
- **Epochs:** 2 (900 total steps)
- **Optimizer:** `paged_adamw_8bit`, lr=1e-4
- **Seed:** 42

**Duration:** ~4 hours 14 minutes on Tesla T4.

**If the Kaggle session times out**, restart the notebook, re-run all cells above training, then run:

```python
trainer.train(resume_from_checkpoint="./epoch-finetuned-qwen3-4b/checkpoint-500")
```

---

## Step 7: Save Adapter

After training completes, the adapter weights and tokenizer are saved automatically to `./qwen3-4B-nli-lora-adapter/` and archived as a zip file.

Pre-trained adapter weights are also available in this repo at `results/models/qwen3-4B-nli-lora-adapter/`.

---

# Inference Notebook (`notebooks/inference.ipynb`)

Runs the fine-tuned model on the test split, applies the playbook, validates evidence spans, and produces all Milestone 1 deliverables (predictions, RunTrace files, evaluation CSV).

## Prerequisites

- Kaggle Notebooks with **GPU T4 x2** accelerator enabled
- The LoRA adapter (either from training above, or from `results/models/qwen3-4B-nli-lora-adapter/`)
- `playbook.yaml` uploaded to Kaggle

## Required Input Paths (must update before running)

These paths are hardcoded in the notebook. Update them to match your Kaggle upload locations:

| Variable (Cell) | Default Path | What to upload |
|-----------------|-------------|----------------|
| `ADAPTER_PATH_OVERRIDE` (Cell 19) | `/kaggle/input/models/harridy/qwen/tensorflow2/default/1/kaggle/working/qwen3-4B-nli-lora-adapter` | Folder with `adapter_config.json` + `adapter_model.safetensors` (from training, or from `results/models/qwen3-4B-nli-lora-adapter/`) |
| `PLAYBOOK_PATH_OVERRIDE` (Cell 35) | `/kaggle/input/datasets/harridy/extras/playbook.yaml` | The `playbook.yaml` file from repo root |

**Optional** (for RunTrace schema validation):

| Path (Cell 41) | Default | What to upload |
|-----------------|---------|----------------|
| Schema file | `/kaggle/input/datasets/harridy/extras/runtrace_ms1.schema.json` | RunTrace JSON schema (skipped if not found) |

## Generated Output Paths

These are created automatically — no configuration needed:

| Variable | Path | Contents |
|----------|------|----------|
| `PREPROCESSED_DIR` | `/kaggle/working/preprocessed/` | Preprocessed train/dev/test JSON splits |
| `CHECKPOINT_FILE` | `/kaggle/working/outputs/checkpoint.json` | Inference progress checkpoint (for crash recovery) |
| `PREDICTIONS_PATH` | `/kaggle/working/outputs/predictions.json` | All predictions with playbook fields |
| `RUNTRACE_DIR` | `/kaggle/working/outputs/runtraces/` | One `runtrace_<contract_id>.json` per contract |
| `EVAL_CSV_PATH` | `/kaggle/working/outputs/evaluation_metrics.csv` | Aggregated evaluation metrics |
| Zip archive | `/kaggle/working/output_files.zip` | All RunTrace files zipped for download |

## Step 1: Install Dependencies & Download Dataset

Open `notebooks/inference.ipynb` on Kaggle with GPU T4 x2 enabled. Run all cells from the top.

```bash
pip install -q json-repair unsloth
```

The dataset downloads automatically via `kagglehub` (same as training notebook). Data exploration and preprocessing cells re-run identically to produce the records needed for evaluation.

## Step 2: Load Model with LoRA Adapter

Continue running cells. The notebook will:

1. Load base model `unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit` in 4-bit on GPU 0.
2. Read `adapter_config.json` from `ADAPTER_PATH_OVERRIDE` and manually attach LoRA weights.
3. Optionally load a second model copy on GPU 1 (if 2 GPUs available) for parallel inference.

## Step 3: Run Inference on Test Split

Continue running cells. The inference loop:

1. Iterates over all 123 test contracts.
2. For each contract, runs all 17 hypotheses through the model.
3. Parses JSON output with retry logic (up to 3 attempts per hypothesis).
4. Saves progress to `/kaggle/working/outputs/checkpoint.json` after each contract — **if the session crashes, re-run this cell and it resumes from where it left off**.

## Step 4: Evidence Validation & Playbook Application

Continue running cells. These steps run automatically:

1. **Evidence span validation** — checks quote integrity (predicted quote matches `contract[char_start:char_end]`) and canonical span overlap.
2. **Playbook mapping** — loads `playbook.yaml` from `PLAYBOOK_PATH_OVERRIDE` and maps each `(hypothesis_id, label)` to deterministic `severity`, `action`, `criticality`, `status`, and `rationale` fields.

## Step 5: Save Outputs

Continue running the remaining cells. The notebook produces:

| Output | Path |
|--------|------|
| Predictions JSON | `/kaggle/working/outputs/predictions.json` |
| RunTrace files (one per contract) | `/kaggle/working/outputs/runtraces/runtrace_<contract_id>.json` |
| Evaluation CSV | `/kaggle/working/outputs/evaluation_metrics.csv` |
| All RunTraces zipped | `/kaggle/working/output_files.zip` |

The evaluation CSV is aggregated from the RunTrace files and contains: `label_accuracy`, `groundedness`, `quote_integrity_pass_rate`, `avg_latency_ms`, `evaluation_timestamp`.

---

## Notes

- Do **not** remove the line `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` — it is required.
- The base model (`Qwen/Qwen3-4B-Instruct-2507`) and dataset splits are fixed across all milestones.
- All random operations use `seed=42` for reproducibility.
- If the inference session crashes, simply re-run the inference cell — checkpoint/resume handles it automatically.
