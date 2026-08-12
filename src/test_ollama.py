from langchain_ollama import ChatOllama


def main():
    model_name = "ibm/granite4.1:3b"

    print(f"Loading model: {model_name}")

    llm = ChatOllama(
        model=model_name,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        keep_alive=0,
    )

    response = llm.invoke(
        "Reply only with: Ollama LangChain connection works."
    )

    print("\nResponse:")
    print(response.content)


if __name__ == "__main__":
    main()