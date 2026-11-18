"""Reproduce the shortest-path/budget mismatch without calling a model."""

import argparse
import json
from collections import deque
from pathlib import Path

DIRECTIONS = (("north", 0, -1), ("east", 1, 0), ("south", 0, 1), ("west", -1, 0))
FIXTURE = Path(__file__).with_name("boxoban_budget_fixture.json")


def shortest_path(board, bound):
    floor = [tuple(cell) for cell in board["floor"]]
    index = {cell: i for i, cell in enumerate(floor)}
    shift = (len(floor) - 1).bit_length()
    mask = (1 << shift) - 1
    adjacent = [[index.get((x + dx, y + dy), -1) for _, dx, dy in DIRECTIONS] for x, y in floor]
    boxes = sum(1 << index[tuple(cell)] for cell in board["boxes"])
    goals = sum(1 << index[tuple(cell)] for cell in board["goals"])
    start = (boxes << shift) | index[tuple(board["player"])]
    frontier = deque([start])
    parent = {start: (None, None)}
    for depth in range(bound + 1):
        for _ in range(len(frontier)):
            state = frontier.popleft()
            boxes, player = state >> shift, state & mask
            if boxes == goals:
                actions = []
                while state != start:
                    state, direction = parent[state]
                    actions.append(DIRECTIONS[direction][0])
                return list(reversed(actions)), len(parent)
            if depth == bound:
                continue
            for direction, target in enumerate(adjacent[player]):
                if target < 0:
                    continue
                moved = boxes
                if boxes & (1 << target):
                    destination = adjacent[target][direction]
                    if destination < 0 or boxes & (1 << destination):
                        continue
                    moved ^= (1 << target) | (1 << destination)
                candidate = (moved << shift) | target
                if candidate not in parent:
                    parent[candidate] = (state, direction)
                    frontier.append(candidate)
    return None, len(parent)


def native_replay(board, actions, budget):
    import random

    import numpy as np
    from omegaconf import OmegaConf

    from balrog.environments import make_env

    config_path = Path(__file__).resolve().parents[2] / "balrog" / "config" / "config.yaml"
    config = OmegaConf.load(config_path)
    config.envs.minihack_kwargs.boxoban_max_episode_steps = budget
    env = make_env("minihack", board["task"], config)
    try:
        random.seed(board["seed"])
        np.random.seed(board["seed"])
        observation, _ = env.reset(seed=board["seed"])
        grid = np.asarray(observation["obs"]["tty_chars"])[1:22, :]

        def coordinates(chars):
            ys, xs = np.where(np.isin(grid, list(map(ord, chars))))
            return sorted([int(x), int(y)] for y, x in zip(ys, xs))

        assert coordinates(".{`@") == board["floor"], "Different map: verify pinned assets and reset order"
        assert coordinates("`") == board["boxes"]
        assert coordinates("{") == board["goals"]
        assert [int(x) for x in observation["obs"]["blstats"][:2]] == board["player"]
        assert list(env.env.language_action_space) == [item[0] for item in DIRECTIONS]
        assert env.max_steps == budget
        for step, action in enumerate(actions, 1):
            _, _, terminated, truncated, _ = env.step(env.check_action_validity(action))
            if terminated or truncated:
                break
        score = float(env.get_stats()["progression"])
        expected = (100, 0.0) if budget == 100 else (104, 1.0)
        assert (step, score) == expected, (step, score, expected)
        assert terminated or truncated
        return {"budget": budget, "steps": step, "progression": score}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="Also replay in the pinned native environment")
    args = parser.parse_args()
    board = json.loads(FIXTURE.read_text())
    shorter, count = shortest_path(board, 100)
    assert shorter is None
    assert count == 796766
    actions, _ = shortest_path(board, 104)
    assert actions is not None and len(actions) == 104
    result = {"states_within_100": count, "minimum_actions": len(actions)}
    if args.native:
        result["native_runs"] = [native_replay(board, actions, budget) for budget in (100, 200, 100, 200)]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
