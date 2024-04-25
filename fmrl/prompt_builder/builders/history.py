from .. import PromptBuilder
from abc import ABC, abstractmethod

class HistoryPromptBuilder(PromptBuilder, ABC):
    def __init__(self, *, max_history=None, max_length=None, action_delim="<|action|>", obs_delim="<|observation|>"):
        self._max_history = max_history
        self._max_length = max_length
        self._obs_history = []
        self._action_history = []
        self._action_delim = action_delim
        self._obs_delim = obs_delim
        
    def update_action(self, action):
        self._action_history.append(action)
        
    def update_observation(self, obs):
        self._obs_history.append(obs)
        
    def reset(self):
        self._obs_history = []
        self._action_history = []
        
    def get_prompt(self):
        n_steps = len(self._action_history)
        start_idx = min(self._max_history or (n_steps + 1), n_steps + 1)
        
        if self._max_length is None:
            return self.format_history(self._obs_history[-start_idx-1:], self._action_history[-start_idx:])
        
        for i in reversed(range(1, start_idx+1)):
            text = self.format_history(self._obs_history[-i-1:], self._action_history[-i:])
            num_tokens = len(text.encode('utf-8')) # Ideal world, we know exactly how many tokens this string is, but estimate using num bytes
            if num_tokens <= self._max_length:
                return text
            
        raise ValueError("Unable to generate context that fits within max_length.")
    
    @abstractmethod
    def format_history(self, obs_history, action_history):
        raise NotImplementedError