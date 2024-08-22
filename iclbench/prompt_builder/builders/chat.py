from .. import PromptBuilder

class ChatPromptBuilder(PromptBuilder):
    def __init__(self, prefix=None):
        if prefix is None:
            prefix = "You are about to be presented with an observation. Respond only with an appropriate action.\n\n"
        self.prefix = prefix
        self._last_obs = None
        
    def update_observation(self, obs):
        self._last_obs = obs
        
    def reset(self):
        self._last_obs = None
        
    def get_prompt(self):
        return [
            {"role": "user", "content": self.prefix + "\n\nObservation:\n\n" + self._last_obs},
        ]
        
    @property
    def is_chat(self):
        return True