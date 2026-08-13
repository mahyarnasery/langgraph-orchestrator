from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


class State(TypedDict):
    question: str
    answer: str


@tool
def get_project_name() -> str:
    """Return the name of the current project."""
    print("\n*** TOOL EXECUTED: get_project_name ***")
    return "langgraph-orchestrator"


def planner_node(state: State):
    print("\n=== Loading model: gemma4:e4b ===")

    llm = ChatOllama(
        model="gemma4:e4b",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        keep_alive=0,
    )

    llm_with_tools = llm.bind_tools([get_project_name])

    # First model call
    messages = [
        HumanMessage(content=state["question"])
    ]

    response = llm_with_tools.invoke(messages)

    # Execute requested tools
    if response.tool_calls:
        for tool_call in response.tool_calls:

            if tool_call["name"] == "get_project_name":

                tool_result = get_project_name.invoke(tool_call["args"])

                messages.append(response)

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

        # Second model call with tool result
        final_response = llm_with_tools.invoke(messages)

        answer = final_response.content

    else:
        answer = response.content

    print("=== Finished model: gemma4:e4b ===")

    return {
        "answer": answer
    }


graph = StateGraph(State)

graph.add_node("planner", planner_node)

graph.set_entry_point("planner")

graph.add_edge("planner", END)

app = graph.compile()


if __name__ == "__main__":

    result = app.invoke(
        {
            "question": (
                "What is the name of the current project? "
                "Use the available tool if needed."
            ),
            "answer": "",
        }
    )

    print("\n\n===== FINAL STATE =====")

    print(result)