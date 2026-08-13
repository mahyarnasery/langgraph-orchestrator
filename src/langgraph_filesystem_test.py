from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


class State(TypedDict):
    request: str
    result: str


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a text file."""
    print(f"\n*** TOOL: read_file({path}) ***")
    return Path(path).read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a text file with the given content."""
    print(f"\n*** TOOL: write_file({path}) ***")
    Path(path).write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace old_text with new_text inside an existing file."""
    print(f"\n*** TOOL: edit_file({path}) ***")

    p = Path(path)
    content = p.read_text(encoding="utf-8")

    if old_text not in content:
        return "OLD_TEXT_NOT_FOUND"

    updated = content.replace(old_text, new_text)

    p.write_text(updated, encoding="utf-8")

    return "EDIT_SUCCESS"


TOOLS = [read_file, write_file, edit_file]


def run_node(state: State):

    llm = ChatOllama(
        model="ibm/granite4.1:3b",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        keep_alive=0,
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [HumanMessage(content=state["request"])]

    # Allow a few tool iterations
    for _ in range(5):

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            return {"result": response.content}

        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool_map = {
                "read_file": read_file,
                "write_file": write_file,
                "edit_file": edit_file,
            }

            tool_result = tool_map[tool_name].invoke(tool_args)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    return {"result": "Too many tool iterations"}


graph = StateGraph(State)

graph.add_node("fs_test", run_node)

graph.set_entry_point("fs_test")

graph.add_edge("fs_test", END)

app = graph.compile()


if __name__ == "__main__":

    request = """
Do these steps in order:

1. Read sandbox/example.txt
2. Create sandbox/new_file.txt containing:
   Created by LangGraph tool test.
3. Edit sandbox/example.txt by replacing:
   Hello from the original file.
   with:
   Hello from the edited file.
4. Tell me exactly what operations succeeded.
"""

    result = app.invoke(
        {
            "request": request,
            "result": "",
        }
    )

    print("\n\n===== FINAL RESULT =====")
    print(result["result"])