from abc import ABC, abstractmethod

from src.orchestration.schemas import TaskBatch, TaskResult


class CodingHarness(ABC):
    """Execution boundary for coding agents."""

    @abstractmethod
    def execute(self, batch: TaskBatch) -> list[TaskResult]:
        """Execute a TaskBatch and return observed task results."""
        raise NotImplementedError