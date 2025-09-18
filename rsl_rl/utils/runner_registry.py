from rsl_rl.runners import OnPolicyRunner

class RunnerRegistry:
    def __init__(self):
        self.runner_classes = {}
    
    def register(self, name: str, runner_class: OnPolicyRunner):
        self.runner_classes[name] = runner_class
    
    def get_runner_class(self, name: str) -> OnPolicyRunner:
        return self.runner_classes[name]

runner_registry = RunnerRegistry()