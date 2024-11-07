class BaseAgent:
    def __init__(self, client_factory, prompt_builder):
        self.client = client_factory()
        self.prompt_builder = prompt_builder

    def act(self, obs):
        raise NotImplementedError

    def update_prompt(self, observation, info, action):
        self.prompt_builder.update_observation(observation, info)
        self.prompt_builder.update_action(action)

    def reset(self):
        self.prompt_builder.reset()
