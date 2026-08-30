class PlannerAgent:
    """Gemma-based architectural planner."""

    def __init__(self, model):
        self.model = model

    def plan(self, objective: str):
        raise NotImplementedError