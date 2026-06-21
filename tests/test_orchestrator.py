# test_orchestrator.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bettercallnli.agent.orchestrator import build_orchestrator
from bettercallnli.agent.history import ConversationHistory

orch = build_orchestrator(retrieval_mode="graphrag")  # or "vector"
print("Orchestrator built OK")
print("Router model:", orch.router.model)
print("ConvAgent model:", orch.conv_agent.model)

# Test that a conversation-mode message routes and runs end-to-end
# (requires a real contract dict)
contract = {"id": "test-001", "text": "This NDA is entered into by Party A and Party B. All confidential information must be kept secret for 5 years."}
history  = ConversationHistory()

result = orch.run(contract, "Who are the parties to this agreement?", history)

print("\n--- Result ---")
print("Mode:     ", result["mode"])           # should be "conversation"
print("Response: ", result["response"][:200])
print("Tool calls:", len(result["tool_calls"]))
print("Router tool call:", result["tool_calls"][0])