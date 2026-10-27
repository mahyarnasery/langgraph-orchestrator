from dataclasses import dataclass


@dataclass(frozen=True)
class AiderConfig:
    model: str = "ollama_chat/ibm/granite4.1:3b"
    edit_format: str = "whole"
    auto_commits: bool = False
    dirty_commits: bool = False
    stream: bool = False
