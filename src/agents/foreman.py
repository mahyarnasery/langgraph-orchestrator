class ForemanAgent:
    """Granite 8B senior reviewer and integrator."""

    def __init__(self, model):
        self.model = model

    def review(self, context):
        raise NotImplementedError

    def integrate(self, context):
        raise NotImplementedError