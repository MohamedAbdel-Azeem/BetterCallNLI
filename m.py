from src.agent.hypothesis_pipeline import HypothesisPipeline
import inspect

methods = [m for m in dir(HypothesisPipeline) if not m.startswith('_')]
print("Public methods:", methods)

# Check the actual signature of run if it exists
if hasattr(HypothesisPipeline, 'run'):
    print("run signature:", inspect.signature(HypothesisPipeline.run))
else:
    print("run() does NOT exist on HypothesisPipeline")