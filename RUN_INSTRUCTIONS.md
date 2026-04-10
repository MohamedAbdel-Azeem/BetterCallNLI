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

## Notes

- Do **not** remove the line `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` — it is required.
- The base model (`Qwen/Qwen3-4B-Instruct-2507`) and dataset splits are fixed across all milestones.
- All random operations use `seed=42` for reproducibility.
