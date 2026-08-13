from typing import TypedDict

import requests
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


class State(TypedDict):
    question: str
    answer: str


@tool
def fetch_page_title(url: str) -> str:
    """Fetch a webpage and return its HTML title."""

    print(f"\n*** TOOL EXECUTED: fetch_page_title({url}) ***")

    response = requests.get(
        url,
        timeout=10,
        headers={
            "User-Agent": "LangGraph-Test/1.0"
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return "No title found"


def planner_node(state: State):

    print("\n=== Loading model: gemma4:e4b ===")

    llm = ChatOllama(
        model="gemma4:e4b",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        keep_alive=0,
    )

    llm_with_tools = llm.bind_tools([fetch_page_title])

    messages = [
        HumanMessage(content=state["question"])
    ]

    # First model call
    response = llm_with_tools.invoke(messages)

    # Execute tools if requested
    if response.tool_calls:

        for tool_call in response.tool_calls:

            if tool_call["name"] == "fetch_page_title":

                tool_result = fetch_page_title.invoke(tool_call["args"])

                messages.append(response)

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

        # Final model call
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
                "Use the fetch_page_title tool to get the title of "
                "https://www.python.org and tell me what the title is."
            ),
            "answer": "",
        }
    )

    print("\n\n===== FINAL STATE =====")

    print(result)