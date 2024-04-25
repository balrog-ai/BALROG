from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from env import NLEExtendedLanguageWrapper
from nle.env import tasks
from tqdm import tqdm
from peft import PeftModel
 
def load(base_model_id, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    # model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
    
    model.resize_token_embeddings(len(tokenizer))
    assert len(tokenizer) == model.get_input_embeddings().weight.shape[0]
    
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    
    return tokenizer, model
 
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
    
    base_model_id = "google/gemma-2b"
    # base_model_id = "google/gemma-7b"
    # base_model_id = "meta-llama/Llama-2-7b-hf"
    checkpoint_path = f"models/{base_model_id}/checkpoint-7800"
    
    tokenizer, model = load(base_model_id, checkpoint_path)
    # model.load_adapter(checkpoint_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20)
    
    env = NLEExtendedLanguageWrapper(base_env, max_length=8000, use_diff_history=False)
    obs = env.reset()
    
    cumreward = 0
    for n_steps in tqdm(range(100000)):
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