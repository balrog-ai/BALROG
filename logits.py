import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

observation = """
You are an agent playing NetHack. This is the current game observation. You are the @ symbol.                                                                                                                
                                                                            
                                                                                
                                                                                
                                                                                
                                     ---------                                  
                                     |.......|                                  
                                     |.......|                                  
                                     +...)..<|                                  
                                     ------.--                                  
                                          ##                                    
                                          #                                     
                                       ####                                     
                                       #                                        
                              #    ----.-----                                   
                              #####.........|                                   
                            #@     |........|                                   
                             ##    ....{....|                                   
                                   ----------                                   
                                                                                
Agent the Hatamoto             St:16 Dx:12 Co:18 In:12 Wi:8 Ch:9 Lawful S:0     
Dlvl:1 $:0 HP:15(15) Pw:2(2) AC:4 Xp:1/0 T:34                               

Possible actions to take:

north
east
south
west
northeast
southeast
southwest
northwest
wait
adjust
apply
cast
close
dip
drop
eat
engrave
enhance
fight
force
jump
kick
pay
pickup
pray
puton
read
search

Output the next action:

"""

model_path = "/scratch0/davide/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
model.eval()


actions = [
    "north",
    "south",
    "west",
    "northeast",
    "southeast",
    "southwest",
    "northwest",
    "wait",
    "adjust",
    "apply",
    "cast",
    "close",
    "dip",
    "drop",
    "eat",
    "engrave",
    "enhance",
    "fight",
    "force",
    "jump",
    "kick",
    "pay",
    "east",
    "pickup",
    "pray",
    "puton",
    "read",
    "search",
]

actions = [
    tokenizer(action, return_tensors="pt").input_ids.to("cuda")[0][1]
    for action in actions
]

print(type(actions[0]))


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


next_action = tokenizer.decode(predict_action(model, tokenizer, observation, actions))
print(next_action)
