import gym
import nle
from nle_language_wrapper import NLELanguageWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")

env = NLELanguageWrapper(gym.make("NetHack-v0"))

action_names = [action_strs[0] for action, action_strs in env.all_nle_action_map.items() if action in env.env.actions]

def get_text_obs(obsv):
    text_obsv = ""
    text_obsv += f"Inventory:\n{obsv['text_inventory']}\n\n"
    text_obsv += f"Stats:\n{obsv['text_blstats']}\n\n"
    text_obsv += f"Cursor:\n{obsv['text_cursor']}\n\n"
    text_obsv += f"Stats:\n{obsv['text_glyphs']}\n\n"
    text_obsv += f"Message:\n{obsv['text_message']}\n\n"
    random.shuffle(action_names)
    text_obsv += f"Output only one of the following actions:\n\n" + ", ".join(action_names) + ".\n\n"
    return text_obsv

obsv = env.reset()

done = False
# while not done:
for i in range(10):
    prompt = get_text_obs(obsv)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(input_ids.keys())
    outputs = model.generate(
        **input_ids,
        # Parameters that control the length of the output
        max_length=10000,       # Adjust this to generate longer sequences
        # Parameters that control the generation strategy
        do_sample=True,
        # Parameters for manipulation of the model output logits
        temperature=10.0,       # Adjust for creativity (lower is more deterministic)
        top_k=50,               # Sample from the top k most likely tokens
        top_p=0.95,             # Use nucleus sampling with this probability threshold
        no_repeat_ngram_size=2, # Prevent repeating n-grams)
        # Parameters that define the output variables of `generate`
        num_return_sequences=1, # Generate 1 sequence (adjust as needed)
    )
    
    print(outputs[0].shape)
    
    print(tokenizer.decode(outputs[0]))
    print(tokenizer.decode(outputs[0]).replace(prompt, ""))
    
    # env.render()
    quit()
    obsv, reward, done, info = env.step(input())
