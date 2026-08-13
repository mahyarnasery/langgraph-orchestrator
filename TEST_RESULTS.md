# LangGraph Tool Calling Validation

Verified on Ubuntu 24.04 LTS with Ollama and `keep_alive=0`.

## Successful models

- `gemma4:e4b`
- `gemma4:e2b`
- `ibm/granite4.1:8b`
- `ibm/granite4.1:3b`

## Verified capabilities

- LangGraph sequential execution
- ChatOllama integration
- Structured `tool_calls`
- LangChain `bind_tools()`
- Tool execution round-trip
- Final response generation after `ToolMessage`
- Automatic VRAM unloading between runs

This confirms that local Ollama models can reliably participate in LangGraph tool-calling workflows on a 4 GB VRAM system when executed sequentially.