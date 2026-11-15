"""Render the date_planner graph structure to a PNG.

Run: .venv\\Scripts\\python.exe graph\\visualize.py
Output: graph\\date_planner_graph.png
"""
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from date_planner import builder

if __name__ == "__main__":
    graph = builder.compile()
    png_bytes = graph.get_graph().draw_mermaid_png()

    out_path = Path(__file__).parent / "date_planner_graph.png"
    out_path.write_bytes(png_bytes)
    print(f"Wrote {out_path}")
