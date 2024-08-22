class BaseAgent:
    def __init__(self, client, prompt_builder):
        self.client = client
        self.prompt_builder = prompt_builder

    def act(self, obs):
        raise NotImplementedError

    def update_prompt(self, observation, action):
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)

    def get_metrics(self):
        return {
            "failed_generation_counter": self.failed_generation_counter,
            "action_history": self.action_history,
            "action_frequency": self.action_frequency,
        }
