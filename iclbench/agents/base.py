class BaseAgent:
    """
    Base class for all agents in the ICLBench framework.

    The `BaseAgent` class serves as a foundational class for creating agents that interact 
    with environments using a client for language model interactions and a prompt builder 
    to manage context. It provides methods for performing actions based on observations, 
    updating the prompt with observations and actions, and resetting the prompt builder.

    Attributes:
        client (Client): An instance of the client created by the client factory.
        prompt_builder (PromptBuilder): An instance of the prompt builder for managing 
                                        prompt context and history.
    """
    
    def __init__(self, client_factory, prompt_builder):
        """
        Initializes the BaseAgent with a client factory and a prompt builder.

        Args:
            client_factory (Callable): A factory function to create a client instance.
            prompt_builder (PromptBuilder): An instance of the prompt builder.
        """
        
        self.client = client_factory()
        self.prompt_builder = prompt_builder

    def act(self, obs):
        """
        Abstract method for performing an action based on the given observation.

        This method must be implemented by subclasses, as the action logic will depend 
        on the specific agent type. For instance, derived agents may store action-observation
        histories when acting.

        Args:
            obs (Observation): The current observation from the environment.
        """
        raise NotImplementedError

    def update_prompt(self, observation, action):
        """
        Updates the prompt builder with the latest observation and action.

        This method incorporates the current observation and the action taken into 
        the prompt history, allowing for context to be maintained across interactions.

        Args:
            observation (Observation): The current observation from the environment.
            action (Action): The action taken by the agent.
        """
    
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)

    def reset(self):
        """
        Resets the prompt builder to its initial state.

        This method clears the prompt history and any other stored information in the 
        prompt builder, allowing for a fresh start in a new episode or task.
        """
        self.prompt_builder.reset()
