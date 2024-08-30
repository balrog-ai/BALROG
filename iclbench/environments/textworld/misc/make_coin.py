import subprocess
import random
import textworld
from concurrent.futures import ProcessPoolExecutor, as_completed


def generate_env(seed, level):
    command = ['python', 'textworld/misc/custom-tw-make.py', 'tw-coin_collector']
    random.seed(seed)
    command += ['--level', level]
    command += ['--format', 'ulx']
    output_path = f"textworld/tw_games/coin_collector/level_{level}_seed_{seed}"
    command += ['--output', output_path]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Command output for seed {seed}:", result.stdout)
        return output_path, seed
    except subprocess.CalledProcessError as e:
        print(f"Error for seed {seed}:", e.stderr)
        return None, seed


def main():
    N = 25
    level = 220
    done = 0
    saved_envs = []

    with ProcessPoolExecutor(max_workers=32) as executor:  # Adjust the number of workers as needed
        futures = {executor.submit(generate_env, random.randint(0, 65535), level): i for i in range(N)}

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
