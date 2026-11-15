from langchain_core.messages import AIMessage
from langgraph.graph import END

from date_planner import route_message


def _state_with_tool_call(name, args=None):
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args or {}, "id": "call_1"}],
            )
        ]
    }


def test_no_tool_calls_routes_to_end():
    state = {"messages": [AIMessage(content="just chatting, no tools needed")]}

    assert route_message(state, config={}) == END


def test_action_tool_call_routes_to_tools_node():
    for tool_name in ("check_weather", "find_places", "web_search"):
        state = _state_with_tool_call(tool_name)
        assert route_message(state, config={}) == "tools"


def test_update_memory_user_routes_to_update_profile():
    state = _state_with_tool_call("UpdateMemory", {"update_type": "user"})

    assert route_message(state, config={}) == "update_profile"


def test_update_memory_date_idea_routes_to_update_date_ideas():
    state = _state_with_tool_call("UpdateMemory", {"update_type": "date_idea"})

    assert route_message(state, config={}) == "update_date_ideas"


def test_update_memory_instructions_routes_to_update_instructions():
    state = _state_with_tool_call("UpdateMemory", {"update_type": "instructions"})

    assert route_message(state, config={}) == "update_instructions"


def test_update_memory_unknown_type_raises():
    state = _state_with_tool_call("UpdateMemory", {"update_type": "bogus"})

    try:
        route_message(state, config={})
        assert False, "expected ValueError"
    except ValueError:
        pass
