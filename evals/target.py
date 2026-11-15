"""Target function(s) invoked by LangSmith `evaluate()` runs.

D1 (route selection) and D2 (tool selection) are evaluated against an
isolated reproduction of what `date_planner()` sends the model - NOT the
compiled `graph` and NOT `date_planner()` itself - because that node also
calls `db.fetch_profile()` / `db.fetch_date_ideas()` (live Postgres) and the
LangGraph store for instructions before it ever reaches the model. The
routing-golden dataset (evals/datasets/routing_golden.py) is deliberately
scoped to an empty profile/date_ideas/instructions case, so this target
reproduces the same system prompt and same tool binding with those left
empty, keeping the D1/D2 CI-gated suite fast, deterministic, and free of any
database dependency. Full-graph runs (with real DB/store state) are for the
nightly/full suite described in evalplan.txt section 6, not this target.

Requires real OPENAI_API_KEY (and TAVILY_API_KEY, since importing tools.py
constructs a TavilySearch client) - this target calls the real model, it is
not mocked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import configuration
from date_planner import MODEL_SYSTEM_MESSAGE, UpdateMemory, model, route_message
from tools import ACTION_TOOLS

_EMPTY_CONFIG = configuration.Configuration()

_ROUTING_MODEL = model.bind_tools(
    [UpdateMemory, *ACTION_TOOLS], parallel_tool_calls=False
)


def _to_message(m: dict):
    role = m["role"]
    if role in ("user", "human"):
        return HumanMessage(content=m["content"])
    if role in ("ai", "assistant"):
        return AIMessage(content=m["content"])
    return SystemMessage(content=m["content"])


def route_decision(inputs: dict) -> dict:
    """LangSmith target for the routing-golden dataset (D1 + D2).

    Expects `inputs["messages"]`: a list of {"role", "content"} dicts making
    up the conversation so far (typically a single user message). Returns
    the route `route_message()` would take plus the raw model tool_calls, so
    D1 and D2 evaluators can both grade the same run.
    """
    messages = [_to_message(m) for m in inputs["messages"]]

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        date_planner_role=_EMPTY_CONFIG.date_planner_role,
        user_profile=None,
        date_ideas="",
        instructions="",
    )

    response = _ROUTING_MODEL.invoke([SystemMessage(content=system_msg)] + messages)

    route = route_message({"messages": messages + [response]}, config={})

    return {
        "route": route,
        "tool_calls": response.tool_calls,
    }
