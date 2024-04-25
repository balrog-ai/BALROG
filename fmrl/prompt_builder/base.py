from abc import ABC, abstractmethod

class PromptBuilder(ABC):
    """
    Abstract class for building prompts for the agent to generate actions.
    """
    def update_action(self, action):
        pass
        
    def update_observation(self, obs):
        pass
        
    def reset(self):
        pass
    
    @abstractmethod
    def get_prompt(self):
        raise NotImplementedError