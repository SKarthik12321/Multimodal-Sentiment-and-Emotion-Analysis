# training/test_import.py
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Calculated project root: {project_root}")

if project_root not in sys.path:
    print(f"Adding project root to sys.path")
    sys.path.insert(0, project_root)
else:
    print("Project root already in sys.path")

print("sys.path:", sys.path)

try:
    from training.models import MultimodalSentimentModel
    print("Successfully imported MultimodalSentimentModel!")

except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

print("If you see this, ALL imports worked!")