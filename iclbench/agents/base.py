class BaseAgent:
    def __init__(self, client, prompt_builder):
        self.client = client
        self.prompt_builder = prompt_builder

    def act(self, obs):
        raise NotImplementedError

    def update_prompt(self, observation, action):
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)
