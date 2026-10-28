from collections.abc import Iterator
from contextlib import contextmanager
import time

from langchain_ollama import ChatOllama


OLLAMA_BASE_URL = "http://127.0.0.1:11434"

PLANNER_MODEL = "gemma4:e4b"
FOREMAN_MODEL = "ibm/granite4.1:8b"
WORKER_MODEL = "ibm/granite4.1:3b"

MAX_INVOCATION_RETRIES = 3


class _ResilientStructuredRunner:
    """
    Wraps a structured-output Runnable to retry both transient Ollama 500
    startup timeouts and empty/malformed JSON responses (OutputParserException).
    """

    def __init__(self, runnable, model_name: str):
        self._runnable = runnable
        self._model_name = model_name

    def invoke(self, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(1, MAX_INVOCATION_RETRIES + 1):
            try:
                res = self._runnable.invoke(*args, **kwargs)
                if res is None:
                    raise ValueError(f"Model {self._model_name} returned None for structured output.")
                return res
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                exc_name = type(exc).__name__.lower()
                retryable = (
                    "timed out waiting for llama-server" in msg
                    or "status code: 500" in msg
                    or "connection" in msg
                    or "invalid json output" in msg
                    or "expecting value" in msg
                    or "outputparserexception" in exc_name
                    or "jsondecodeerror" in exc_name
                    or "returned none" in msg
                )
                if attempt < MAX_INVOCATION_RETRIES and retryable:
                    print(
                        f"[WARN] Transient error from {self._model_name} "
                        f"({type(exc).__name__}, attempt {attempt}/{MAX_INVOCATION_RETRIES}). Retrying in 2s..."
                    )
                    time.sleep(2)
                    continue
                raise
        raise last_exc  # type: ignore[misc]


class _ResilientChatOllama:
    """Thin wrapper around ChatOllama that adds retry logic to structured output calls."""

    def __init__(self, llm: ChatOllama, model_name: str):
        self._llm = llm
        self._model_name = model_name

    def with_structured_output(self, schema, **kwargs):
        runnable = self._llm.with_structured_output(schema, **kwargs)
        return _ResilientStructuredRunner(runnable, self._model_name)

    def __getattr__(self, name: str):
        return getattr(self._llm, name)


@contextmanager
def model_session(model_name: str) -> Iterator[_ResilientChatOllama]:
    """Create a short-lived Ollama model session with automatic retry."""
    print(f"\n=== Loading model: {model_name} ===")

    llm = ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        keep_alive=0,
        timeout=180,
    )

    try:
        yield _ResilientChatOllama(llm, model_name)
    finally:
        print(f"=== Finished model: {model_name} ===")
