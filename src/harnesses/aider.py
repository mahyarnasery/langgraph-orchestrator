from typing import Any

from src.harnesses.base import CodingHarness


class AiderHarness(CodingHarness):
    """Aider-backed coding execution harness."""

    def execute(self, task: Any) -> Any:
        raise NotImplementedError("Aider integration will be implemented later.")