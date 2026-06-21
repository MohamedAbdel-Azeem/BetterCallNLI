# test_pipeline.py
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bettercallnli.agent.history import ConversationHistory
from bettercallnli.agent.orchestrator import build_orchestrator

# ── load dev.json ──────────────────────────────────────────────────────────
DEV_JSON_PATH = str(_REPO_ROOT / "data" / "dev.json")

with open(DEV_JSON_PATH, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

raw = dev_data["documents"][0]

contract = {
    "id":   str(raw["id"]),
    "text": raw["text"],
}

print(f"Loaded contract: {contract['id']}  ({len(contract['text'])} chars)")

# ── ground truth ───────────────────────────────────────────────────────────
NDA_TO_H = {v: k for k, v in {
    "H01": "nda-1",  "H02": "nda-2",  "H03": "nda-3",
    "H04": "nda-4",  "H05": "nda-5",  "H06": "nda-7",
    "H07": "nda-8",  "H08": "nda-10", "H09": "nda-11",
    "H10": "nda-12", "H11": "nda-13", "H12": "nda-15",
    "H13": "nda-16", "H14": "nda-17", "H15": "nda-18",
    "H16": "nda-19", "H17": "nda-20",
}.items()}

ground_truth = {}
for nda_id, ann in raw["annotation_sets"][0]["annotations"].items():
    h_id = NDA_TO_H.get(nda_id)
    if h_id:
        ground_truth[h_id] = ann["choice"]

_NORM = {
    "Entailment":    "ENTAILED",
    "Contradiction": "CONTRADICTED",
    "NotMentioned":  "NOT_MENTIONED",
}

# ── run ────────────────────────────────────────────────────────────────────
orchestrator = build_orchestrator(retrieval_mode="graphrag")
history      = ConversationHistory()

# swap the message below to test conversation vs hypothesis mode:
#   "analyze this contract"                   → hypothesis_analysis
#   "What are the confidentiality obligations?" → conversation
MESSAGE = "What are the confidentiality obligations?"

print(f"\nMessage: '{MESSAGE}'")
print("Running — this will take a few minutes for hypothesis mode...\n")

result = orchestrator.run(contract, MESSAGE, history)

# ── FIX 1: branch on mode before accessing verdicts ───────────────────────
if result["mode"] == "hypothesis_analysis":
    print(f"{'HYP':<5} {'PREDICTED':<15} {'GROUND TRUTH':<15} {'MATCH':<6} {'CONF':<6} {'EV'}")
    print("-" * 65)

    correct = 0
    for v in result["verdicts"]:
        h_id      = v["hypothesis_id"]
        predicted = v["label"]
        gt_raw    = ground_truth.get(h_id, "N/A")
        gt        = _NORM.get(gt_raw, gt_raw)
        match     = "✅" if predicted == gt else "❌"
        conf      = f"{v['confidence']:.2f}"
        ev        = len(v.get("evidence", []))
        if predicted == gt:
            correct += 1
        print(f"{h_id:<5} {predicted:<15} {gt:<15} {match:<6} {conf:<6} {ev}")

    total = len(result["verdicts"])
    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Tool calls fired: {len(result['tool_calls'])}")
    print(f"Agent traces:     {len(result['agent_traces'])}")

    print("\nRetry breakdown:")
    for trace in result["agent_traces"]:
        flag     = " ← needed retries" if trace["attempts"] > 1 else ""
        accepted = "✅" if trace["accepted"] else "❌ (best-effort)"
        print(f"  {trace['hypothesis_id']}  attempts={trace['attempts']}  {accepted}{flag}")

    # ── FIX 2: reviewer score inspection ──────────────────────────────────
    print("\nReviewer scores per hypothesis:")
    for trace in result["agent_traces"]:
        h_id = trace["hypothesis_id"]
        reviewer_calls = [
            tc for tc in trace["tool_calls"]
            if tc["name"] == "hypothesis_reviewer"
        ]
        for rc in reviewer_calls:
            out = rc["output"]
            print(
                f"  {h_id}  attempt={rc['args']['attempt']}"
                f"  score={out['score']}/10"
                f"  accepted={out['accepted']}"
                f"  label_alignment={out['label_alignment']}"
                f"  evidence_quality={out['evidence_quality']}"
                f"  reasoning={out['reasoning_coherence']}"
            )
            if out.get("critique"):
                print(f"    critique: {out['critique'][:120]}")

else:
    print("Mode: conversation")
    print("Response:", result["response"][:500])
    print(f"\nTool calls fired: {len(result['tool_calls'])}")