import sys
from pathlib import Path

# Dummy creds so importing graph modules doesn't fail at construction time
# (ChatOpenAI/TavilySearch validate a key is *present*, they don't call out
# to it at import time). Real calls are mocked or never exercised in tests.
import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
os.environ.setdefault("TAVILY_API_KEY", "test-dummy-key")

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
