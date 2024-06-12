import os
import pickle
from fmrl.environments.nle.utils import text_render
from fmrl.prompt_builder import ConcatHistoryPromptBuilder
from generate_episodes import nle_action_textmap

def load_episode(path, game_id):
    with open(os.path.join(path, f"{game_id}_summary.pkl"), "rb") as file:
        summary = pickle.load(file)
        
    with open(os.path.join(path, f"{game_id}_data.pkl"), "rb") as file:
        data = pickle.load(file)
        
    return summary, data

def postprocess(summary, timesteps):
    # TODO: different prompting strategies here
    prompt_builder = ConcatHistoryPromptBuilder(
        max_length=8000,
    )
    
    samples = []
    
    for timestep in timesteps:
        # TODO: different observation rendering strategies here
        obs = text_render(timestep)
        action = timestep["action"]
        
        prompt_builder.update_observation(obs)
        prompt_builder.update_action(action)

        samples.append({"prompt": prompt_builder.get_prompt(), "completion": nle_action_textmap[action]})

    return samples

def load_dataset(path, game_ids):
    samples = []
    
    for game_id in game_ids:
        summary, data = load_episode(path, game_id)
        samples.extend(postprocess(summary, data))

    with open("samples.pkl", "wb") as file:
        pickle.dump(samples, file)
    
if __name__=="__main__":
    load_dataset("data", [1,])