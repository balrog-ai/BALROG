from gym import Wrapper

class PromptBuilderWrapper(Wrapper):
    def __init__(self, env, prompt_builder=None):
        super().__init__(env)
        self.prompt_builder = prompt_builder
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.prompt_builder.update_action(action)
        self.prompt_builder.update_observation(observation)
        info["prompt"] = self.prompt_builder.build_prompt()
        return observation, reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        self.prompt_builder.reset()
        self.prompt_builder.update_observation(observation)
        return observation