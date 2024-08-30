import subprocess
import random


def make_cooking_game(seed, recipe, take, go, open_container, cook, cut, output_path):
    # Start building the command with the basic call
    command = ['tw-make', 'tw-cooking']

    # Add the recipe and take options
    command += ['--recipe', str(recipe), '--take', str(take)]

    # Add the go option
    command += ['--go', str(go)]

    # Add flags based on boolean variables
    if open_container:
        command.append('--open')
    if cook:
        command.append('--cook')
    if cut:
        command.append('--cut')

    # Fixed setting for --split
    command += ['--split', 'test']

    # Add the output path
    command += ['--output', output_path]

    # Print the command to be executed (for verification)
    print('Executing command:', ' '.join(command))

    # Execute the command
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Command output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)


N = 25
for i in range(N):
    seed = random.randint(0, 25565)
    recipe = 5
    go = 12
    open_container = True
    cook = True
    cut = True
    make_cooking_game(
        seed=seed,
        recipe=recipe,  # Number of ingredients in the recipe
        take=recipe,  # Number of ingredients to find
        go=go,  # Number of locations
        open_container=open_container,  # Whether to open containers
        cook=cook,  # Whether to cook some ingredients
        cut=cut,  # Whether to cut ingredients
        output_path=f"textworld/tw_games/the_cooking_game/cooking_item{recipe}_seed_{seed}"  # Path to save the game
    )
