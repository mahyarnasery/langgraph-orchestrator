from collections.abc import Iterator
from contextlib import contextmanager
import subprocess
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


def _build_ollama_client(
    model_name: str,
    *,
    temperature: float = 0.0,
    force_cpu: bool = False,
) -> ChatOllama:
    """
    Create a ChatOllama instance tuned for model stability:
    - Avoid overriding `num_ctx` on `gemma4` models so Ollama does not split
      Gemma 4 PLE tensors across GPU/CPU (`GGML_SCHED_MAX_SPLIT_INPUTS`).
    - Support `force_cpu=True` (`num_gpu=0`) as an automatic fallback if
      `GGML_SCHED_MAX_SPLIT_INPUTS` is ever triggered by VRAM fragmentation.
    """
    extra_kwargs: dict = {}
    if "gemma4" not in model_name.lower():
        extra_kwargs["num_ctx"] = 4096
    if force_cpu:
        extra_kwargs["num_gpu"] = 0

    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        keep_alive=0,
        timeout=180,
        **extra_kwargs,
    )


class _ResilientStructuredRunner:
    """
    Wraps a structured-output Runnable to retry transient Ollama 500 startup
    timeouts, GGML_SCHED_MAX_SPLIT_INPUTS hardware split errors, and empty JSON
    responses (OutputParserException).
    """

    def __init__(self, llm: ChatOllama, schema, model_name: str, kwargs: dict):
        self._llm = llm
        self._schema = schema
        self._model_name = model_name
        self._kwargs = kwargs
        self._runnable = llm.with_structured_output(schema, **kwargs)

    def invoke(self, prompt_or_input, *args, **kwargs):
        last_exc: Exception | None = None
        use_cpu_fallback = False

        for attempt in range(1, MAX_INVOCATION_RETRIES + 1):
            try:
                active_runnable = self._runnable
                active_input = prompt_or_input

                if attempt > 1:
                    retry_llm = _build_ollama_client(
                        self._model_name,
                        temperature=round(0.15 * (attempt - 1), 2),
                        force_cpu=use_cpu_fallback,
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
                is_ggml_split_err = (
                    "ggml_assert" in msg
                    or "ggml_sched_max_split_inputs" in msg
                    or "llama-server process has terminated" in msg
                )
                if is_ggml_split_err or "status code: 500" in msg:
                    # Retry on GPU on Attempt 2 after stopping and letting VRAM clear;
                    # only fall back to slow CPU (num_gpu=0) on Attempt 3 (attempt >= 2).
                    use_cpu_fallback = (attempt >= 2)
                    subprocess.run(
                        ["ollama", "stop", self._model_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    time.sleep(1.0)

                retryable = (
                    is_ggml_split_err
                    or "timed out waiting for llama-server" in msg
                    or "status code: 500" in msg
                    or "connection" in msg
                    or "invalid json output" in msg
                    or "expecting value" in msg
                    or "outputparserexception" in exc_name
                    or "jsondecodeerror" in exc_name
                    or "returned none" in msg
                )
                if attempt < MAX_INVOCATION_RETRIES and retryable:
                    mode_note = " (with CPU fallback num_gpu=0)" if use_cpu_fallback else ""
                    print(
                        f"[WARN] Transient error from {self._model_name} "
                        f"({type(exc).__name__}, attempt {attempt}/{MAX_INVOCATION_RETRIES}). "
                        f"Retrying in 2s{mode_note}..."
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
    """Create a short-lived Ollama model session with clean VRAM handoff."""
    print(f"\n=== Loading model: {model_name} ===")

    llm = _build_ollama_client(model_name, temperature=0.0)

    try:
        yield _ResilientChatOllama(llm, model_name)
    finally:
        subprocess.run(
            ["ollama", "stop", model_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(0.6)
        print(f"=== Finished model: {model_name} ===")
