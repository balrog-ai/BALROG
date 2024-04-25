from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from env import NLEExtendedLanguageWrapper
from const import SIMPLE_ACTIONS
from nle.env import tasks
from tqdm import tqdm
import torch
from typing import List
from peft import PeftModel

def load(base_model_id, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
    
    model.resize_token_embeddings(len(tokenizer))
    assert len(tokenizer) == model.get_input_embeddings().weight.shape[0]
    
    model = PeftModel.from_pretrained(model, model_id)
    model = model.merge_and_unload()
    
    return tokenizer, model
 
def evaluate_model(tokenizer, model, max_steps=100000):
    @torch.no_grad()
    def predict_action(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        observation: str,
        actions: List[torch.Tensor],
    ) -> str:
        """
        Predict the next action using logits of the model

        Args:
        model (AutoModelForCausalLM): The model to use for prediction
        tokenizer (AutoTokenizer): The tokenizer to use for prediction
        observation (str): The game observation
        actions (List[str]): The list of possible actions to take

        Returns:

        str: The predicted action
        """
        inputs = tokenizer(observation, return_tensors="pt").input_ids.to("cuda")
        outputs = model(inputs)
        logits = outputs.logits

        # This are the logits of the last token of the input
        last_token_logits = logits[0, -1, :]

        # Get the logit position of each action token
        action_logits = {action: last_token_logits[action] for action in actions}
        # Get the action with the highest logit
        best_action_token = max(action_logits, key=action_logits.get)
        return best_action_token
    
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

    actions = [
        tokenizer(action, return_tensors="pt").input_ids.to("cuda")[0][1]
        for action in SIMPLE_ACTIONS
    ]
    
    env = NLEExtendedLanguageWrapper(base_env, max_length=8000, use_diff_history=False)
    obs = env.reset()
    
    cumreward = 0
    for n_steps in tqdm(range(max_steps)):
        action = tokenizer.decode(predict_action(model, tokenizer, obs, actions))
        print(action)
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
    
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
    
    base_model_id = "google/gemma-2b"
    # base_model_id = "google/gemma-7b"
    # base_model_id = "meta-llama/Llama-2-7b-hf"
    checkpoint_path = f"models/{base_model_id}/checkpoint-8000"
    
    tokenizer, model = load(base_model_id, checkpoint_path)
    
    evaluate_model(tokenizer, model)