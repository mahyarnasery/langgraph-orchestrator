from contextlib import contextmanager
from collections.abc import Iterator

from langchain_ollama import ChatOllama


OLLAMA_BASE_URL = "http://127.0.0.1:11434"

PLANNER_MODEL = "gemma4:e4b"
FOREMAN_MODEL = "ibm/granite4.1:8b"
WORKER_MODEL = "ibm/granite4.1:3b"


@contextmanager
def model_session(model_name: str) -> Iterator[ChatOllama]:
    """
    Create a short-lived Ollama model session.

    keep_alive=0 is intentional: the model should not remain resident
    after the invocation completes.
    """

    print(f"\n=== Loading model: {model_name} ===")

    llm = ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        keep_alive=0,
    )

    try:
        yield llm

    finally:
        print(f"=== Finished model: {model_name} ===")