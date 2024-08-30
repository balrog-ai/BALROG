import subprocess
import random
import textworld
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

infos = textworld.EnvInfos(
    feedback=True,  # Response from the game after typing a text command.
    description=True,  # Text describing the room the player is currently in.
    inventory=True,  # Text describing the player's inventory.
    command_templates=True,
    admissible_commands=True,
    location=True,
    moves=True,
    won=True,
)


def generate_env(seed):
    command = ['python', 'textworld/misc/custom-tw-make.py', 'tw-treasure_hunter']
    command += ['--level', '30']
    command += ['--format', 'ulx']
    output_path = f"textworld/tw_games/treasure_hunter/seed_{seed}"
    command += ['--output', output_path]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Command output for seed {seed}:", result.stdout)
        env = textworld.start(output_path + '.ulx', request_infos=infos)
        game_state = env.reset()
        if len(game_state['extra.walkthrough']) == 20:
            print(f"seed {seed}: demo length {len(game_state['extra.walkthrough'])}. SAVED!")
            return output_path, seed
        else:
            files = [output_path + '.ulx', output_path + '.ni', output_path + '.json']
            for f in files:
                if os.path.exists(f):
                    os.remove(f)
            return None, seed
    except subprocess.CalledProcessError as e:
        print(f"Error for seed {seed}:", e.stderr)
        return None, seed


def main():
    N = 25
    done = 0
    saved_envs = []

    with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust the number of workers as needed
        futures = {executor.submit(generate_env, random.randint(0, 65535)): i for i in range(N)}

        for future in as_completed(futures):
            output_path, seed = future.result()
            if output_path:
                saved_envs.append(output_path)
                done += 1
                if done >= N:
                    break

    print(f"Collected {done} environments.")


if __name__ == "__main__":
    main()
