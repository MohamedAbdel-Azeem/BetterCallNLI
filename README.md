# BetterCallNLI

NDA document classification and review system that analyzes contracts using Natural Language Inference.
## Notebook Links
Model: https://www.kaggle.com/code/belalabdelazeem/qwen3-4b-unsloth
Inference: https://www.kaggle.com/code/harridy/notebook8c78216aee
## Repository Structure

```
BetterCallNLI/
├── README.md                  # Project overview and run instructions
├── CONTRIBUTION.md            # Group member contributions
├── notebooks/                 # Model/training/inference notebooks (annotated)
├── src/                       # Modular source code
├── playbooks/                 # Playbook files used in evaluation
├── results/                   # Final evaluation CSV and RunTrace files
├── data/                      # Datasets or instructions to obtain them
└── requirements.txt           # Python dependencies
```

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| a | Model/training/inference code | `notebooks/`, `src/` |
| b | Reproducible run instructions | Annotations within notebooks |
| c | Final evaluation CSV and model weights | `results/` |
| d | RunTrace file (with base-model info) | `results/` |
| e | Playbook file used in evaluation | `playbooks/` |
| f | Contribution file | `CONTRIBUTION.md` |

## Milestone 1: Baseline Model

### Dataset — ContractNLI

The dataset was sourced from the [ContractNLI Kaggle mirror](https://www.kaggle.com/datasets/seiftarek158/contract-nli) and loaded via `kagglehub`. It contains 607 NDA documents split into three sets:

| Split | Documents | (Doc, Hypothesis) Pairs |
|-------|-----------|------------------------|
| Train | 423       | 7,191                  |
| Dev   | 61        | 1,037                  |
| Test  | 123       | 2,091                  |

Each document is evaluated against a fixed set of 17 hypotheses (nda-1 through nda-20, with some IDs unused), covering clauses such as *Limited use*, *No reverse engineering*, *Sharing with employees*, *Survival of obligations*, and more.

---

### Data Engineering & Preparation

#### Label Distribution

Across the 7,191 training pairs, the label distribution is moderately imbalanced:

| Label | Count | Percentage |
|-------|-------|------------|
| ENTAILED | 3,530 | 49.1% |
| NOT_MENTIONED | 2,820 | 39.2% |
| CONTRADICTED | 841 | 11.7% |

CONTRADICTED is the minority class overall. However, label skew varies heavily by hypothesis — for example, `nda-4` (Limited use) is ENTAILED in 86.1% of contracts, while `nda-11` (No reverse engineering) is NOT_MENTIONED in 85.8%.

#### Evidence Span Coverage

60.8% of (doc, hypothesis) pairs carry at least one evidence span. All 2,820 zero-evidence pairs correspond exclusively to NOT_MENTIONED labels, which is structurally expected since there is no contract text to cite when a clause is absent.

#### Contract Length & Token Budget

Contracts range from 1,481 to 54,571 characters (mean ~11,049 chars, median ~9,936 chars). Using the approximation of 4 characters per token for English legal text:

| Context Window | Contracts Fitting |
|----------------|-------------------|
| 1,024 tokens | 9.5% (40 / 423) |
| 2,048 tokens | 37.6% (159 / 423) |
| 4,096 tokens | 83.7% (354 / 423) |
| 8,192 tokens | 99.1% (419 / 423) |

Evidence spans themselves are fairly compact, with a median length of ~50 tokens and a mean of ~63 tokens across 8,341 annotated spans in the training set.

#### Evidence Position Analysis

An analysis of where evidence spans appear within each contract (measured as the relative character position 0–1) revealed a clear front-loading bias:

| Contract Decile | % of Evidence Spans |
|-----------------|---------------------|
| 0–10% | 5.6% |
| 10–20% | 18.2% |
| 20–30% | 17.9% |
| 30–40% | 16.3% |
| 40–50% | 13.5% |
| 50–60% | 11.6% |
| 60–70% | 7.8% |
| 70–80% | 5.1% |
| 80–90% | 2.3% |
| 90–100% | 1.7% |

The first 10% of each contract contributes only 5.6% of evidence spans — a strong signal that the very opening of contracts (typically just a title and parties section) contains little substantive clause text. This directly informed the truncation strategy.

#### Truncation Strategy

To fit contracts within the model's context window at training time, the following strategy was adopted:

- **Skip the first 10%** of each contract (catches only 5.6% of evidence spans, not worth the token cost).
- **Take up to 1,400 tokens** (5,600 chars) of the remaining text for training prompts (reserving ~400 tokens for the JSON output).
- **Take up to 1,700 tokens** (6,800 chars) for inference prompts (no output budget needed).
- Append a `[TRUNCATED]` marker when the contract is cut short.

Under this strategy, 70.4% of training records and 78.7% of dev records required truncation. Evidence spans outside the visible window are excluded from training targets for those records.

#### Prompt Format

Each training example is formatted as a JSON-constrained generation prompt:

```
Classify the hypothesis based on the contract. Respond with ONLY valid JSON nothing else:
{"label": "ENTAILED" | "CONTRADICTED" | "NOT_MENTIONED", "evidence": ["exact quote 1", "exact quote 2"]}
If label is NOT_MENTIONED, evidence must be [].
Evidence must be copied verbatim from the contract text, word for word. Do not paraphrase or invent.
Contract:
<contract text>

Hypothesis:
<hypothesis text>

{"label": "ENTAILED", "evidence": ["..."]}
```

The model is fine-tuned to produce a single JSON object as output, with verbatim evidence quotes copied from the contract.

---
 
### Model Selection Criteria & Rationale
 
Two candidate models were evaluated before settling on a final base model for the project:
 
- **`Qwen/Qwen3-4B-Instruct-2507`**
- **`meta-llama/Llama-3.2-3B-Instruct`**
 
Both were fine-tuned on the ContractNLI training split under identical conditions. Qwen3-4B consistently produced better-structured JSON outputs, higher label accuracy on the dev set, and cleaner evidence span extraction. Llama-3.2-3B-Instruct showed competitive tunability but lagged behind on the structured generation requirements of this task. **Qwen3-4B-Instruct-2507 was therefore selected as the fixed base model for all milestones.**
 
This choice is further backed by independent third-party benchmarking conducted by [distil labs](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/), which evaluated 12 SLMs across 8 diverse tasks (classification, document understanding, and question answering) after fine-tuning with identical hyperparameters.
 
#### Post-Fine-Tuning Performance Ranking
 
![Model Performance Ranking After Fine-Tuning](https://www.distillabs.ai/_astro/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning-01_model_ranking_plot.CDeUX0On_Z2povqd.webp)
 
*Source: distil labs — [We Benchmarked 12 SLMs Across 8 Tasks](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/)*
 
Results are aggregated as average rank across all benchmarks (lower is better), with 95% confidence intervals:
 
| Model | Average Rank | 95% CI |
|-------|-------------|--------|
| **Qwen3-4B-Instruct-2507** | **2.25** | ±1.03 |
| Qwen3-8B | 2.75 | ±1.37 |
| Llama-3.1-8B-Instruct | 4.00 | ±1.42 |
| Qwen3-1.7B | 4.44 | ±1.60 |
| **Llama-3.2-3B-Instruct** | **4.56** | ±1.73 |
| Qwen3-0.6B | 5.11 | ±1.86 |
 
Qwen3-4B-Instruct-2507 ranks **#1** overall after fine-tuning, outperforming even the larger Qwen3-8B. Llama-3.2-3B-Instruct ranks **#5**, confirming the performance gap observed in our own experiments.
 
The Qwen3 family dominates the top 4 spots. Notably, Qwen3-4B's July 2025 update (`-2507`) delivers better fine-tuned results than the larger 8B variant, suggesting architectural improvements that are particularly well-suited for knowledge distillation and structured output tasks.

---

### Base Model Selection & QLoRA Fine-tuning

#### Base Model

The base model selected for Milestone 1 (and fixed for all subsequent milestones) is:

**`Qwen/Qwen3-4B-Instruct-2507`**

This model was chosen for its strong instruction-following capability, compact size suitable for Kaggle's Tesla T4 GPU (14.56 GB VRAM), and native support for structured JSON generation. It was loaded through [Unsloth](https://github.com/unslothai/unsloth)'s `FastLanguageModel` interface, which applies kernel-level optimizations for up to 2x faster fine-tuning on consumer GPUs.

#### 4-bit Quantization

The model was loaded in 4-bit NF4 quantization using `BitsAndBytesConfig`:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

Double quantization further compresses the quantization constants themselves, reducing memory footprint without meaningful accuracy loss.

#### LoRA Adapter Configuration

LoRA adapters were applied to the attention projection layers only:

| Parameter | Value |
|-----------|-------|
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Rank (`r`) | 12 |
| Alpha (`lora_alpha`) | 32 |
| Dropout | 0.15 |
| Trainable parameters | 8,847,360 |
| Total model parameters | 4,031,315,456 |
| Trainable % | **0.22%** |

Only 0.22% of the model's weights are updated during training, keeping memory and compute requirements low while still adapting the model to the legal NLI task.

#### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 2 |
| Per-device batch size | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Learning rate | 1e-4 |
| Optimizer | `paged_adamw_8bit` |
| Max sequence length | 2,048 |
| Gradient checkpointing | Enabled |
| Mixed precision | Disabled (fp16=False, bf16=False) |
| Evaluation strategy | Every 20% of an epoch |
| Seed | 42 |
| Total steps | 900 |

Training ran for approximately **4 hours and 14 minutes** on a single Tesla T4. Training was resumed from checkpoint-500 due to the session time constraint.

#### Training Results

| Step | Training Loss | Validation Loss |
|------|--------------|-----------------|
| 540 | 0.977 | 1.478 |
| 720 | 0.883 | 1.555 |
| 900 | 0.862 | 1.587 |

Final training loss: **0.862**. The gap between training and validation loss is expected given the relatively small dev set (1,037 records) and the generative nature of the task (evidence quotes are open-ended).

#### Pre- vs Post-Training Output

Before fine-tuning, the base model did not reliably produce valid JSON — outputting free-text responses like:

```
Classification:
"NOT_MENTIONED"
Evidence: []
```

After fine-tuning, the model consistently produces well-formed JSON with correct labels and verbatim evidence spans:

```json
{"label": "ENTAILED", "evidence": ["The foregoing shall not prevent either party from disclosing Information which is:", "iii) rightfully received from a third party; or"]}
```

#### Adapter Artifact

The LoRA adapter weights and tokenizer were saved to `./qwen3-4B-nli-lora-adapter/` and archived as `qwen-nli-lora-adapter.zip` for portability across sessions.

---

## Base Model

**`Qwen/Qwen3-4B-Instruct-2507`** — fine-tuned with QLoRA (r=12, α=32) on ContractNLI for 2 epochs.