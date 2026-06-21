"""
Local inference client — drop-in replacement for huggingface_hub.InferenceClient.

Loads a BASE HuggingFace model (e.g. Qwen/Qwen2.5-7B-Instruct) and exposes the
same `chat_completion(...)` signature the BetterCallNLI agents already call,
so the entire Router → Analyst → Reviewer pipeline can run against a model
loaded locally on a Kaggle GPU — no HuggingFace Serverless API calls.

Spec context (MS3 §2f)
----------------------
The fine-tuned MS1 LoRA adapter MUST NOT be loaded here. Use only a same-family
base model (e.g. Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct). The wrapper does
not enforce this — it will happily load anything HF gives it — so callers are
responsible for passing a base-model `model_name` only.

Why this exists
---------------
The HF Serverless API has rate limits and shared-quota latency. Running the
full ~123-contract × 17-hypothesis evaluation against the agentic pipeline
makes ~6 000+ LLM calls; doing them locally on a Kaggle T4 sidesteps any
quota issues at the cost of slightly slower per-call generation.

Usage on Kaggle
---------------
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",    # BASE — no adapter
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Install the shim BEFORE building the orchestrator so every agent
    # constructor picks up the local client.
    from src.agent.local_inference_client import install_as_global_client
    install_as_global_client(model, tokenizer)

    from src.agent.orchestrator import build_orchestrator
    orch = build_orchestrator(retrieval_mode="vector")
    # All LLM calls from this orchestrator now run on the local GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── lightweight HF-shaped return objects ──────────────────────────────────────

@dataclass
class _ChatMessage:
    role: str
    content: str


@dataclass
class _ChatChoice:
    index: int
    message: _ChatMessage
    finish_reason: str


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _ChatCompletionOutput:
    """Mimics huggingface_hub.ChatCompletionOutput shape."""
    id: str
    model: str
    choices: List[_ChatChoice]
    usage: _Usage


# ── the client ────────────────────────────────────────────────────────────────

class LocalInferenceClient:
    """
    Mimics the subset of `huggingface_hub.InferenceClient` that the
    BetterCallNLI agents actually call: `chat_completion(messages, max_tokens,
    temperature, ...)`.

    Constructor accepts `(model=…, token=…)` so the agents can call us with
    their existing kwargs; the `token` arg is harmlessly ignored.
    `chat_completion` returns a `_ChatCompletionOutput` whose
    `choices[0].message.content` and `usage.{prompt_tokens, completion_tokens}`
    exactly match the HF API shape the agents read.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        token: Optional[str] = None,
        device: str = "cuda",
        **_ignored: Any,
    ) -> None:
        if model is None or tokenizer is None:
            raise ValueError(
                "LocalInferenceClient requires `model` and `tokenizer`. "
                "Call install_as_global_client(model, tokenizer) BEFORE "
                "constructing the orchestrator."
            )
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

        # Agents read `self.model` for runtrace metadata; mirror that
        self.model = getattr(model, "name_or_path", "local-base-model")

    # ── public API ────────────────────────────────────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.4,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        **_extra: Any,
    ) -> _ChatCompletionOutput:
        """Generate one completion. Returns a HF-shaped output dataclass."""
        import torch  # local import — only required when client is actually called

        prompt = self._render_prompt(messages)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if self._device:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])

        do_sample = temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample":      do_sample,
            "top_p":          top_p,
            "pad_token_id":   self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_token_ids = output_ids[0, prompt_tokens:]
        completion_tokens = int(new_token_ids.shape[0])
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        # Honour user-supplied stop sequences (best-effort)
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0]

        return _ChatCompletionOutput(
            id="local-0",
            model=self.model,
            choices=[
                _ChatChoice(
                    index=0,
                    message=_ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=_Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ── prompt rendering ──────────────────────────────────────────────────────

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Render chat messages → single prompt string using the tokenizer's chat
        template when available, otherwise a generic ChatML-style fallback.
        """
        apply = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply) and getattr(self._tokenizer, "chat_template", None):
            try:
                return apply(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Manual ChatML fallback (works for Qwen2/3 family)
        lines: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)


# ── global install helper ─────────────────────────────────────────────────────

_AGENT_MODULES = (
    "src.agent.intent_router",
    "src.agent.conversation_agent",
    "src.agent.hypothesis_analyst",
    "src.agent.reviewer_agent",
)


def install_as_global_client(model: Any, tokenizer: Any, device: str = "cuda") -> None:
    """
    Replace `huggingface_hub.InferenceClient` so every agent that does
    `InferenceClient(model=…, token=…)` gets a LocalInferenceClient instead.

    This patches:
      - the canonical symbol on `huggingface_hub` (for modules imported AFTER this call)
      - the already-bound symbol on every BetterCallNLI agent module that did
        `from huggingface_hub import InferenceClient` (for modules imported BEFORE)

    Call this BEFORE constructing the Orchestrator on Kaggle.
    """
    import sys
    import huggingface_hub

    captured_model = model
    captured_tokenizer = tokenizer
    captured_device = device

    class _PatchedInferenceClient(LocalInferenceClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs = {k: v for k, v in kwargs.items() if k not in {"model", "tokenizer"}}
            super().__init__(
                model=captured_model,
                tokenizer=captured_tokenizer,
                device=captured_device,
                **kwargs,
            )

    # 1. Patch the canonical symbol on the huggingface_hub module
    huggingface_hub.InferenceClient = _PatchedInferenceClient

    # 2. Patch every already-imported agent module that bound InferenceClient locally
    for mod_name in _AGENT_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "InferenceClient"):
            mod.InferenceClient = _PatchedInferenceClient
