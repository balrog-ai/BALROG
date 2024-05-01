import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nle.env import tasks
from fmrl.environments import NLETextWrapper, NLEAsciiWrapper
from fmrl.prompt_builder import ChatPromptBuilder
import argparse
from nle.nethack import ACTIONS
from nle_language_wrapper import NLELanguageWrapper
from render import tty_render_image, tty_render_image_action_history
from PIL import Image
import wandb
import imageio
import glob
import matplotlib.pyplot as plt

ACTION_NAMES = [action_strs[0] for action, action_strs in NLELanguageWrapper.all_nle_action_map.items() if action in ACTIONS]
ACTIONS_LIST_STR = ",\n".join(ACTION_NAMES)
INSTRUCTION_PROMPT = f"""
You are an agent playing NetHack. In a moment I will present you an observation. Only output an action from the following list:
{ACTIONS_LIST_STR}.

For example, a valid output would simply be "{ACTION_NAMES[0]}" or "{ACTION_NAMES[1]}".
You can only output one action at a time. The goal is to maximize the reward.
Don't just output the example actions above, output the action that you think will maximize the reward.
""".strip()

# def plot_action_bar(action_counter):
#     fig, ax = plt.subplots()
#     ax.bar(action_counter.keys(), action_counter.values())
#     ax.set_xticklabels(action_counter.keys(), rotation=45)
#     ax.set_xlabel("Action")
#     ax.set_ylabel("Count")
#     plt.tight_layout()
#     plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-2b-it")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--savedir", type=str, default=None)
    parser.add_argument("--num_tries", type=int, default=10, help="Number of return sequences from the model / number of attempts model gets to generate valid actions per timestep.")
    parser.add_argument("--obs_style", choices=["ascii_map", "language"], default="language")
    parser.add_argument("--prompt_builder_strategy", choices=["simple", "chat"], default="chat")
    args = parser.parse_args()
    
    env = tasks.NetHackChallenge(
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
                "tty_colors",
            ),
            penalty_step=0.0,
            penalty_time=0.0,
            penalty_mode="constant",
            no_progress_timeout=100,
            # save_ttyrec_every=1,
        )
    )
    
    # Set how language observations are represented
    if args.obs_style == "ascii_map":
        env = NLEAsciiWrapper(env)
    elif args.obs_style == "language":
        env = NLETextWrapper(env)
    else:
        raise ValueError(f"Unknown obs_style: {args.obs_style}")
    
    # Set how prompts are built
    if args.prompt_builder_strategy == "simple":
        raise NotImplementedError
    elif args.prompt_builder_strategy == "chat":
        prompt_builder = ChatPromptBuilder(INSTRUCTION_PROMPT)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2)
    generate_kwargs = {
        # "max_length": 100,
        "temperature": 0.8,
        # "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": args.num_tries,
        # "no_repeat_ngram_size": 3,
        "do_sample": True,
    }
    
    obs = env.reset()
    prompt_builder.update_observation(obs["prompt"])
    
    wandb.login()
    wandb.init(project="nle-language-model-test", config=args)
    
    if args.savedir and not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    cumreward = 0
    failed_generation_counter = 0
    action_counter = {action: 0 for action in ACTION_NAMES}
    action_history = []
    
    for step in range(args.max_steps):
        with open(f"./outputs/observations.txt", "a") as f:
            f.write(f"======================\nOBSERVATION (t={step})\n======================\n\n" + obs["prompt"] + "\n\n")
        prompt = prompt_builder.get_prompt()
        outputs = generator(prompt, return_full_text=False, **generate_kwargs)
        # outputs = [{"generated_text": input()},] * args.num_tries
        for i, output in enumerate(outputs):
            action = output["generated_text"]
            try:
                obs, reward, done, info = env.step(action)
                break
            except:
                pass
        if i == len(outputs) - 1:
            print("Failed to generate a valid action. Defaulting to \"esc\".")
            obs, reward, done, info = env.step("esc")
            failed_generation_counter += 1
            continue
        action_counter[action] += 1
        action_history.append(action)
        if args.savedir is not None:
            tty_image = Image.fromarray(tty_render_image_action_history(obs["tty_chars"], obs["tty_colors"], action_history))
            tty_image.save(f"{args.savedir}/{step:09}.png")
        prompt_builder.update_action(action)
        prompt_builder.update_observation(obs["prompt"])        
        cumreward += reward
        wandb.log({
            "cumreward": cumreward,
            # "image": wandb.Image(tty_image),
            "action_counter": wandb.plot.bar(wandb.Table(data=list(action_counter.items()), columns=["action", "count"]), "action", "count", title="Action Counts"),
        })
        if done:
            break
        
    # generate gif
    if args.savedir is not None:
        images = []
        for image_file in glob.glob(os.path.join(args.savedir, "*.png")):
            image = imageio.imread(image_file)
            images.append(image)
        gif_path = os.path.join(args.savedir, "animation.gif")
        imageio.mimsave(gif_path, images, duration=0.2)