from collections import defaultdict


class BaseAgent:
    def __init__(self):
        self.failed_generation_counter = 0
        self.action_history = []
        self.action_frequency = defaultdict(int)

    def act(self, obs):
        raise NotImplementedError

    def update_prompt(self, observation, action):
        raise NotImplementedError

    def get_metrics(self):
        return {
            "failed_generation_counter": self.failed_generation_counter,
            "action_history": self.action_history,
            "action_frequency": self.action_frequency,
        }
