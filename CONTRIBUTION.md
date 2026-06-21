# Contributions M3

### Group Members

| Name | Contributions |
|------|---------------|
| Omar Elharridy | HypothesisAnalyst agent (structured verdict with label, confidence, evidence spans, counter-evidence, and retry-on-feedback), HypothesisPipeline orchestration (Analyst → Reviewer loop with best-candidate fallback), local inference support via Transformers pipeline, and initial M3 codebase scaffold. |
| Abdelrahman Wael | IntentRouter (keyword fast-path + LLM-based routing); Orchestrator (single entry point);   — per-prompt intent routing via IntentRouter beside session-level mode lock;ASCII Saul Goodman |
| Mohamed Abdelazeem | ReviewerAgent (3-dimension rubric: label alignment, evidence quality, reasoning coherence; scored 1–10 with structured critique fed back to the Analyst on retry), KAA pipeline integration, and review-threshold configuration. |
| Yahia Hesham | PlaybookEnricher (deterministic policy layer: status/severity/action/rationale mapping from playbook.yaml), RuntraceFormatter (schema-compliant runtrace emission for both contract-mode and conversation-mode), GraphRAG retriever upgrade to hybrid pipeline with auto-hypothesis routing, and Kaggle bundler fixes. |
| Seif Tarek | MS3 CLI with Rich-based terminal UI (verdict cards, evidence display, animated spinners), evaluation runner, single-file Kaggle bundle + bundler script, LocalInferenceClient shim for Kaggle GPU inference, and evaluation sharding + per-contract checkpointing for parallel runs. |

# Contributions M1

### Group Members

| Name | Contributions |
|------|---------------|
| Seif Tarek | Dataset preparation & preprocessing, train/dev/eval split definition, and initial prompt writing. |
| Mohamed Khaled & Yahia Hesham | Base LLM selection, QLoRA fine-tuning, hyperparameter tuning & iterative prompt enhancement|
| Omar Elharridy | Inference pipeline over all 17 hypotheses, output parsing, evidence span extraction, and quote integrity validation. |
| Abdelrahman Wael | Playbook mapping engine, RunTrace JSON generation, and final metrics CSV. |