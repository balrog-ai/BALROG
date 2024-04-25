from .. import PromptBuilder

class DefaultPromptBuilder(PromptBuilder):
    def __init__(self):
        self._last_obs = None
        
    def update_observation(self, obs):
        self._last_obs = obs
    
    def reset(self):
        self._last_obs = None
        
    def get_prompt(self):
        return self._last_obs