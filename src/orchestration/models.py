from collections.abc import Iterator
from contextlib import contextmanager
import time

from langchain_ollama import ChatOllama


OLLAMA_BASE_URL = "http://127.0.0.1:11434"

PLANNER_MODEL = "gemma4:e4b"
FOREMAN_MODEL = "ibm/granite4.1:8b"
WORKER_MODEL = "ibm/granite4.1:3b"

MAX_INVOCATION_RETRIES = 3


def configure_models(
    *,
    planner_model: str | None = None,
    foreman_model: str | None = None,
    worker_model: str | None = None,
) -> None:
    """Override active Ollama model identifiers at runtime."""
    global PLANNER_MODEL, FOREMAN_MODEL, WORKER_MODEL
    if planner_model and planner_model.strip():
        PLANNER_MODEL = planner_model.strip()
    if foreman_model and foreman_model.strip():
        FOREMAN_MODEL = foreman_model.strip()
    if worker_model and worker_model.strip():
        WORKER_MODEL = worker_model.strip()


def get_planner_model() -> str:
    return PLANNER_MODEL


def get_foreman_model() -> str:
    return FOREMAN_MODEL


def get_worker_model() -> str:
    return WORKER_MODEL


class _ResilientStructuredRunner:
    """
    Wraps a structured-output Runnable to retry transient Ollama 500 startup
    timeouts and empty/malformed JSON responses (OutputParserException).
    On retry after OutputParserException, bumps temperature slightly and appends
    a strict JSON instruction so temperature=0 never repeats an empty response.
    """

    def __init__(self, llm: ChatOllama, schema, model_name: str, kwargs: dict):
        self._llm = llm
        self._schema = schema
        self._model_name = model_name
        self._kwargs = kwargs
        self._runnable = llm.with_structured_output(schema, **kwargs)

    def invoke(self, prompt_or_input, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(1, MAX_INVOCATION_RETRIES + 1):
            try:
                active_runnable = self._runnable
                active_input = prompt_or_input

                if attempt > 1:
                    retry_llm = ChatOllama(
                        model=self._model_name,
                        base_url=OLLAMA_BASE_URL,
                        temperature=round(0.15 * (attempt - 1), 2),
                        keep_alive=0,
                        timeout=180,
                    )
                    active_runnable = retry_llm.with_structured_output(
                        self._schema, **self._kwargs
                    )
                    if isinstance(prompt_or_input, str):
                        active_input = (
                            prompt_or_input.strip()
                            + "\n\nCRITICAL: Respond ONLY with a valid, non-empty JSON object matching the required schema."
                        )

                res = active_runnable.invoke(active_input, *args, **kwargs)
                if res is None:
                    raise ValueError(
                        f"Model {self._model_name} returned None for structured output."
                    )
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
                        f"({type(exc).__name__}, attempt {attempt}/{MAX_INVOCATION_RETRIES}). "
                        f"Retrying in 2s with adaptive sampling..."
                    )
                    time.sleep(2)
                    continue
                raise
        raise last_exc  # type: ignore[misc]


class _ResilientChatOllama:
    """Thin wrapper around ChatOllama that adds adaptive retry logic to structured output calls."""

    def __init__(self, llm: ChatOllama, model_name: str):
        self._llm = llm
        self._model_name = model_name

    def with_structured_output(self, schema, **kwargs):
        return _ResilientStructuredRunner(
            self._llm, schema, self._model_name, kwargs
        )

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
