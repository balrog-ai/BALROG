import gym
from autoascend.env_wrapper import EnvWrapper
from render import tty_render_image
from PIL import Image

def rollout():
    env = EnvWrapper(
        gym.make("NetHackChallenge-v0", no_progress_timeout=100),
        agent_args=dict(panic_on_errors=True, verbose=False),
        step_limit=10000000000,
    )
    
    env.main()
    data = env.get_data()

    for t in range(100):
        datum = data[t]
        img = tty_render_image(datum["tty_chars"], datum["tty_colors"])
        Image.fromarray(img).save(f"tmp/img_{t}.png")

if __name__ == "__main__":
    rollout()