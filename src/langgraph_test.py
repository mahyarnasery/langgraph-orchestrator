from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


class State(TypedDict):
    message: str
    planner_output: str
    worker_output: str


def call_model(model_name: str, prompt: str) -> str:
    print(f"\n=== Loading model: {model_name} ===")

    llm = ChatOllama(
        model=model_name,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        keep_alive=0,
    )

    response = llm.invoke(prompt)

    print(f"=== Finished model: {model_name} ===")

    return response.content


def planner_node(state: State):
    output = call_model(
        "gemma4:e4b",
        f"""
You are a planner.

User request:
{state['message']}

Give a short plan.
""",
    )

    return {
        "planner_output": output
    }


def worker_node(state: State):
    output = call_model(
        "ibm/granite4.1:3b",
        f"""
You are a worker.

Follow this plan:

{state['planner_output']}

Explain what you would do.
""",
    )

    return {
        "worker_output": output
    }


graph = StateGraph(State)

graph.add_node("planner", planner_node)
graph.add_node("worker", worker_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "worker")
graph.add_edge("worker", END)

app = graph.compile()


if __name__ == "__main__":

    result = app.invoke(
        {
            "message": "Build a simple Python web scraper",
            "planner_output": "",
            "worker_output": "",
        }
    )

    print("\n\n===== FINAL STATE =====")

    print(result)