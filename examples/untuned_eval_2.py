from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from env import NLEExtendedLanguageWrapper
from nle.env import tasks
from tqdm import tqdm
from prompt_builder import PromptBuilder
 
class ChatPromptBuilder(PromptBuilder):
    def __init__(self):
        self._last_obs = None
        
    def append_observation(self, obs):
        self._last_obs = obs
        
    def reset(self):
        self._last_obs = None
        
    def get_prompt(self):
        return [
            {"role": "user", "content": "You are an agent playing NetHack. In a moment I will present you observations. Only output an action from the following list:\n" + ",\n".join(ACTION_NAMES) + "\n\n"},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": self._last_obs},
        ]
 
if __name__=="__main__":    
    base_env = tasks.NetHackChallenge(
        **dict(
            # savedir="./experiment_outputs/dummy_ttyrec",
            character="@",
            max_episode_steps=100000000,
            observation_keys=(
                "blstats",
                "tty_chars",
                "tty_cursor",
                "glyphs",
                "inv_strs",
                "inv_letters",
            ),
            penalty_step=0.0,
            penalty_time=0.0,
            penalty_mode="constant",
            no_progress_timeout=100,
            # save_ttyrec_every=1,
        )
    )
    
    model_id = "google/gemma-2b-it"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20)
    
    prompt_builder = ChatPromptBuilder()
    
    env = NLEExtendedLanguageWrapper(base_env, max_length=8000, prompt_builder=prompt_builder)
    obs = env.reset()
    
    cumreward = 0
    for n_steps in tqdm(range(100000)):
        breakpoint()
        action = generator(obs, return_full_text=False)
        obs, reward, done, info = env.step(action)
        cumreward += reward
        if done:
            break
    
    with open("output.txt", "w") as text_file:
        text_file.write(env.history())
    print(f"Total reward: {cumreward}")
    print(f"Progress: {info['progress']}")
    print(f"Achievements: {info['achievements']}")
    print(f"Highest Achievements: {info['highest_achievement']}")