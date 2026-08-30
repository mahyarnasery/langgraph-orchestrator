from src.harnesses.base import CodingHarness
from src.orchestration.schemas import TaskBatch, TaskResult


class WorkerAgent:
    """Implementation agent executed through a coding harness."""

    def __init__(self, harness: CodingHarness):
        self.harness = harness

    def run(self, batch: TaskBatch) -> list[TaskResult]:
        return self.harness.execute(batch)