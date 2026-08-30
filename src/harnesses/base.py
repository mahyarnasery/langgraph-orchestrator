from abc import ABC, abstractmethod
from typing import Any


class CodingHarness(ABC):
    """Interface for a coding-agent execution harness."""

    @abstractmethod
    def execute(self, task: Any) -> Any:
        """Execute a coding task and return the result."""
        raise NotImplementedError