class WorkerAgent:
    """Granite 3B implementation agent."""

    def __init__(self, model, harness):
        self.model = model
        self.harness = harness

    def execute(self, task):
        raise NotImplementedError