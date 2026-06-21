# test_router.py  (put it at the project root)
import os
from dotenv import load_dotenv
load_dotenv()

from src.agent.intent_router import IntentRouter

router = IntentRouter(hf_token=os.environ["HF_TOKEN"])

test_cases = [
    # should be "conversation"
    "What are the confidentiality obligations?",
    "Does this NDA allow sharing with employees?",
    "Summarise the termination clause",
    "Who are the parties to this agreement?",
    # should be "hypothesis_analysis"
    "Analyze this contract",
    "Run a full review",
    "Check all hypotheses",
    "Generate the NDA report",
    "Run the full hypothesis analysis",
]

for msg in test_cases:
    intent, tool_call = router.route(msg)
    source = "keyword" if "keyword" in tool_call["output"]["raw_response"] else "llm"
    print(f"[{intent:25s}] ({source:7s})  {msg}")