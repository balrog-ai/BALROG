from fmrl.environments.nle import NLELanguageWrapper
import nle
import gym
from PIL import Image

env = gym.make("NetHackChallenge-v0")
env = NLELanguageWrapper(env, prompt_mode="hybrid")
obs = env.reset()
image = Image.fromarray(env.render("tty_image")).save("1tty_image.png")
image = Image.fromarray(env.render("image")).save("2image.png")
print(obs["text"])