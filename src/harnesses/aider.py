from src.harnesses.base import CodingHarness
from src.orchestration.schemas import TaskBatch, TaskResult


class AiderHarness(CodingHarness):
    """Aider-backed coding execution harness."""

    def execute(self, batch: TaskBatch) -> list[TaskResult]:
        raise NotImplementedError(
            "AiderHarness execution is not implemented yet."
        )