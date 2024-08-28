from gym import Wrapper

BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]


class BabyAITextCleanLangWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.language_action_space = BABYAI_ACTION_SPACE[:]
        self._last_mission = None

    @property
    def interleaving_token(self):
        return self._interleaving_token

    def get_prompt(self, obs, infos):
        def _form_prompt(description):
            return "\n".join([d.replace("You see ", "") for d in description])

        prompt = _form_prompt(infos["descriptions"])
        return prompt

    def reset(self):
        obs, infos = self.env.reset()
        prompt = self.get_prompt(obs, infos)
        self._mission = obs["mission"]
        # Following the convention from NetHack Language Wrapper for specifying 
        # short term vs long term context here. There is no equivalent long term 
        # context like e.g. inventory in BabyAI-Text. 
        obs['text'] = (prompt, "") 
        return obs

    def step(self, action):
        action_int = self.language_action_space.index(action)
        obs, reward, done, infos = self.env.step(action_int)
        prompt = self.get_prompt(obs, infos)
        obs['text'] = (prompt, "")
        return obs, reward, done, infos

    def get_stats(self):
        # No special stats tracking implemented for now
        return {'mission': self._mission}
