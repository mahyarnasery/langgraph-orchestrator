from langchain_ollama import ChatOllama
import time

models = [
    "ibm/granite4.1:3b",
    "ibm/granite4.1:8b",
    "gemma4:e2b",
]

for model in models:
    print(f"\n=== {model} ===")

    start = time.time()

    llm = ChatOllama(
        model=model,
        base_url="http://127.0.0.1:11434",
        keep_alive=0,
        timeout=120,
    )

    response = llm.invoke("Reply with OK")

    elapsed = time.time() - start

    print(f"Time: {elapsed:.1f}s")
    print(response.content)

    time.sleep(2)