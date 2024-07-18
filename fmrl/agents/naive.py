import requests
import wandb
import logging
from copy import deepcopy

MAX_STEPS_IN_EPISODE = int(1000000)

class NaiveAgent(object):
    def __init__(self, make_env, url, make_prompt_builder, *, default_payload = None):
        self.env = make_env()
        self.url = url
        self.prompt_builder = make_prompt_builder()
        self.default_payload = default_payload
        
        # some collected statistics
        self.failed_generation_counter = 0
        self.action_history = []
        self.episode_return = 0.
        self.action_frequency = {action: 0 for action in self.env.language_action_space}
        
    def act(self):
        payload = deepcopy(self.default_payload)
        
        input = self.prompt_builder.get_prompt()
        using_chat = isinstance(input, list)
        
        # chat is handled differently
        if using_chat:
            payload["messages"] = input
        else:
            payload["text"] = input
        
        response = requests.post(self.url, json=payload)
        # response.raise_for_status()
        if response.status_code != 200:
            raise Exception(response.json()["message"])

        response = response.json()
        
        action = None
        if using_chat:
            for choice in response["choices"]:
                candidate_action = choice["message"]["content"]
                if candidate_action in self.env.language_action_space:
                    action = candidate_action
                    break
        else:
            for choice in response["choices"]:
                candidate_action = choice["text"]
                if candidate_action in self.env.language_action_space:
                    action = candidate_action
                    break
            
        if action:
            self.action_frequency[action] += 1
            self.action_history.append(action)
            return action
        
        action = self.env.language_action_space.sample()
        logging.warn(f'Failed to generate a valid action. Randomly selecting action \"{action}\".')
        self.failed_generation_counter += 1
        return action
        
    def run(self):
        obs = self.env.reset()
        self.prompt_builder.update_observation(obs["text"])
        cumreward = 0.
        
        for _ in range(MAX_STEPS_IN_EPISODE):
            action = self.act()
            obs, reward, done, info = self.env.step(action)
            self.prompt_builder.update_action(action)
            self.prompt_builder.update_observation(obs["text"])
            self.episode_return += reward
            if done:
                break
        
        return cumreward
        
    def get_metrics(self):
        return {
            "episode_return": self.episode_return,
            "failed_generation_counter": self.failed_generation_counter,
            "action_history": self.action_history,
            "action_frequency": self.action_frequency,
        }