import sys
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

# `langgraph dev` loads this file directly by path and does not add its
# directory to sys.path, so the bare `import configuration` below would
# otherwise fail depending on how this module gets loaded.
sys.path.insert(0, str(Path(__file__).parent))
import configuration
import db

## Utilities

# Inspect the tool calls for Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Extract information from tool calls for both patches and new memories in Trustcall
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "DateIdea", "Profile")
    """
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                if call['args']['patches']:
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'value': call['args']['patches'][0]['value']
                    })
                else:
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits']
                    })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f"Document {change['doc_id']} unchanged:\n"
                f"{change['planned_edits']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts)

## Schema definitions

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with, for the purpose of planning dates."""
    name: Optional[str] = Field(description="The user's name", default=None)
    address: Optional[str] = Field(description="The user's address", default=None)
    partner_name: Optional[str] = Field(description="The name of the person the user is dating/planning dates for", default=None)
    budget: Optional[str] = Field(description="The user's budget preference for dates", default=None)
    interests: list[str] = Field(
        description="Interests and hobbies the user or their partner has, relevant to date planning",
        default_factory=list
    )
    dietary_restrictions: list[str] = Field(
        description="Any dietary restrictions or food preferences to keep in mind",
        default_factory=list
    )

# Date idea schema
class DateIdea(BaseModel):
    idea: str = Field(description="The date idea or activity")
    location: Optional[str] = Field(description="Suggested venue, neighborhood, or location", default=None)
    estimated_cost: Optional[str] = Field(description="Estimated cost or budget tier (e.g. '$', '$$', '$100')", default=None)
    date_time: Optional[datetime] = Field(
        description="When the date is planned for, if scheduled",
        default=None
    )
    notes: list[str] = Field(
        description="Specific, actionable details (e.g. restaurant names, booking links, backup plans)",
        min_items=1,
        default_factory=list
    )
    status: Literal["idea", "planned", "booked", "completed", "archived"] = Field(
        description="Current status of the date idea",
        default="idea"
    )

## Initialize the model and tools

from tools import ACTION_TOOLS

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'date_idea', 'instructions']

# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

## Create the Trustcall extractor for updating the user profile
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

## Prompts

# Chatbot instruction for choosing what to update and what tools to call
MODEL_SYSTEM_MESSAGE = """{date_planner_role}

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them, their partner, and preferences)
2. The user's list of date ideas
3. General instructions for updating the date idea list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current list of Date Ideas (may be empty if none have been added yet):
<date_ideas>
{date_ideas}
</date_ideas>

Here are the current user-specified preferences for updating the date idea list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

You also have tools for real-world information, which you can call directly.
Route strictly by topic - never fall back to web_search for something a
dedicated tool below already covers, even if web_search could technically
answer it too:
- Singapore weather / forecast / rain / temperature -> check_weather. Pass hours_ahead matching how far out the date actually is - only reach for a larger horizon when the date is genuinely that far off. Use this before suggesting outdoor date ideas, or whenever weather is relevant.
- Singapore PSI / haze / air quality -> check_psi. Use this before suggesting outdoor or strenuous activities if haze could be a concern.
- Singapore UV index / sun exposure -> check_uv_index. Use this before suggesting extended outdoor activities during the day.
- Real venues (restaurants, activities) -> find_places. Use this to ground date ideas in actual, real places rather than inventing plausible-sounding ones.
- Anything else current or specific not covered above (event listings, opening hours, reviews) -> web_search.

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below.

2. Decide whether any of your long-term memory should be updated:
- If personal information was provided about the user or their partner, update the user's profile by calling UpdateMemory tool with type `user`
- If a date idea, plan, or activity is mentioned, update the date idea list by calling UpdateMemory tool with type `date_idea`
- If the user has specified preferences for how to update the date idea list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Decide whether a real-world tool call (check_weather, check_psi, check_uv_index, find_places, web_search) would make your answer more useful or concrete. Prefer calling a tool over guessing or inventing specifics like venue names, addresses, weather, or air quality. Use web_search only for things the other tools don't cover.

4. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user when you update the date idea list
- Do not tell the user that you have updated instructions

5. Err on the side of saving new date ideas. No need to ask for explicit permission.

6. Respond naturally to the user after a tool call was made, or if no tool call was made."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

Use the provided tools to retain any necessary memories about the user.

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the date idea list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update the date idea list. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

## Node definitions

def date_planner(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load the profile and date ideas from Supabase, and instructions from the
    store, and use them to personalize the chatbot's response."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    date_category = configurable.date_category
    date_planner_role = configurable.date_planner_role

    # Retrieve profile from Supabase
    user_profile = db.fetch_profile(user_id, date_category)

    # Retrieve date ideas from Supabase
    date_idea_rows = db.fetch_date_ideas(user_id, date_category)
    date_ideas = "\n".join(f"{row}" for row in date_idea_rows)

    # Retrieve custom instructions
    namespace = ("instructions", date_category, user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        date_planner_role=date_planner_role,
        user_profile=user_profile,
        date_ideas=date_ideas,
        instructions=instructions,
    )

    response = model.bind_tools([UpdateMemory, *ACTION_TOOLS], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig):

    """Reflect on the chat history and update the user's profile in Supabase."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    date_category = configurable.date_category

    existing_profile = db.fetch_profile(user_id, date_category)
    existing_memories = [("profile", "Profile", existing_profile)] if existing_profile else None

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    result = profile_extractor.invoke({"messages": updated_messages,
                                         "existing": existing_memories})

    profile = result["responses"][0]
    db.upsert_profile(user_id, date_category, profile.model_dump(mode="json"))

    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_calls[0]['id']}]}

def update_date_ideas(state: MessagesState, config: RunnableConfig):

    """Reflect on the chat history and update the user's date ideas in Supabase."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    date_category = configurable.date_category

    tool_name = "DateIdea"

    existing_rows = db.fetch_date_ideas(user_id, date_category)
    existing_memories = [
        (str(row["id"]), tool_name, {k: v for k, v in row.items() if k != "id"})
        for row in existing_rows
    ] or None

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    spy = Spy()

    date_idea_extractor = create_extractor(
        model,
        tools=[DateIdea],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    result = date_idea_extractor.invoke({"messages": updated_messages,
                                         "existing": existing_memories})

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        row_id = rmeta.get("json_doc_id", str(uuid.uuid4()))
        db.upsert_date_idea(row_id, user_id, date_category, r.model_dump(mode="json"))

    tool_calls = state['messages'][-1].tool_calls

    date_idea_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": date_idea_update_msg, "tool_call_id": tool_calls[0]['id']}]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the instructions memory."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    date_category = configurable.date_category

    namespace = ("instructions", date_category, user_id)

    existing_memory = store.get(namespace, "user_instructions")

    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    new_memory = model.invoke([SystemMessage(content=system_msg)] + state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])

    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id": tool_calls[0]['id']}]}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig) -> Literal[END, "update_date_ideas", "update_instructions", "update_profile", "tools"]:

    """Reflect on the memories and chat history to decide whether to update the
    memory collection or call one of the real-world action tools."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END

    tool_call = message.tool_calls[0]
    tool_name = tool_call['name']

    if tool_name != "UpdateMemory":
        # check_weather / find_places / web_search
        return "tools"

    if tool_call['args']['update_type'] == "user":
        return "update_profile"
    elif tool_call['args']['update_type'] == "date_idea":
        return "update_date_ideas"
    elif tool_call['args']['update_type'] == "instructions":
        return "update_instructions"
    else:
        raise ValueError

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

builder.add_node(date_planner)
builder.add_node(update_date_ideas)
builder.add_node(update_profile)
builder.add_node(update_instructions)
builder.add_node("tools", ToolNode(ACTION_TOOLS))

builder.add_edge(START, "date_planner")
builder.add_conditional_edges("date_planner", route_message)
builder.add_edge("update_date_ideas", "date_planner")
builder.add_edge("update_profile", "date_planner")
builder.add_edge("update_instructions", "date_planner")
builder.add_edge("tools", "date_planner")

# Compile the graph — no checkpointer/store passed here: the LangGraph
# platform (both `langgraph dev` locally and LangGraph Cloud in prod)
# injects persistence automatically based on deployment config.
graph = builder.compile()
