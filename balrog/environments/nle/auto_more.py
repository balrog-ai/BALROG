import gymnasium as gym
from nle import nle_language_obsv
from nle.nethack import actions as A


class AutoMore(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.nle_language = nle_language_obsv.NLELanguageObsv()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["text_message"] = self.nle_language.text_message(obs["tty_chars"]).decode("latin-1")

        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)

        message = self.nle_language.text_message(obs["tty_chars"]).decode("latin-1")
        done = term or trun
        while "--More--" in message and not done:
            message = message.replace("--More--", "\n")

            action_index = self.env.actions.index(A.MiscAction.MORE)
            obs, rew, term, trun, info = self.env.step(action_index)
            done = term or trun
            add = self.nle_language.text_message(obs["tty_chars"]).decode("latin-1")
            message += add
            reward += rew

        obs["text_message"] = message

        return obs, reward, term, trun, info
