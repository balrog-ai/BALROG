import atexit
import contextlib
import gc
import multiprocessing
import os
import shutil
import sys
import tempfile
import termios
import tty

from pathlib import Path
from pprint import pprint

import nle.nethack as nh

from autoascend import (
    agent as agent_lib,
)  # the library can be reloaded in `reload_agent` function
from autoascend.visualization import visualizer


def fork_with_nethack_env(env):
    tmpdir = tempfile.mkdtemp(prefix="nlecopy_")
    shutil.copytree(env.env._vardir, tmpdir, dirs_exist_ok=True)
    env.env._tempdir = None  # it has to be done before the fork to avoid removing the same directory two times
    gc.collect()

    pid = os.fork()

    env.env._tempdir = tempfile.TemporaryDirectory(prefix="nlefork_")
    shutil.copytree(tmpdir, env.env._tempdir.name, dirs_exist_ok=True)
    env.env._vardir = env.env._tempdir.name
    os.chdir(env.env._vardir)
    return pid


def reload_agent(base_path=str(Path(__file__).parent.absolute())):
    global visualize, agent_lib
    visualize = agent_lib = None
    modules_to_remove = []
    for k, m in sys.modules.items():
        if hasattr(m, "__file__") and m.__file__ and m.__file__.startswith(base_path):
            modules_to_remove.append(k)
    del m

    gc.collect()
    while modules_to_remove:
        for k in modules_to_remove:
            assert sys.getrefcount(sys.modules[k]) >= 2
            if sys.getrefcount(sys.modules[k]) == 2:
                sys.modules.pop(k)
                modules_to_remove.remove(k)
                gc.collect()
                break
        else:
            assert 0, (
                "cannot unload agent library",
                {k: sys.getrefcount(sys.modules[k]) for k in modules_to_remove},
            )


class ReloadAgent(KeyboardInterrupt):
    # it inherits from KeyboardInterrupt as the agent doesn't catch that exception
    pass


class EnvWrapper:
    def __init__(
        self,
        env,
        to_skip=0,
        visualizer_args=dict(enable=False),
        step_limit=None,
        agent_args={},
        interactive=False,
    ):
        self.env = env
        self.agent_args = agent_args
        self.interactive = interactive
        self.to_skip = to_skip
        self.step_limit = step_limit
        self.visualizer = None
        if visualizer_args["enable"]:
            visualizer_args.pop("enable")
            self.visualizer = visualizer.Visualizer(self, **visualizer_args)
        self.last_observation = None
        self.agent = None

        self.draw_walkable = False
        self.draw_seen = False
        self.draw_shop = False

        self.is_done = False

    def _init_agent(self):
        self.agent = agent_lib.Agent(self, **self.agent_args)

    def main(self):
        self.reset()
        while 1:
            try:
                self._init_agent()
                self.agent.main()
                break
            except ReloadAgent:
                pass
            finally:
                self.render()

            self.agent = None
            reload_agent()

    def reset(self):
        obs = self.env.reset()
        self.score = 0
        self.step_count = 0
        self.end_reason = ""
        self.last_observation = obs
        self.is_done = False

        if self.agent is not None:
            self.render()

        agent_lib.G.assert_map(obs["glyphs"], obs["chars"])

        blstats = agent_lib.BLStats(*obs["blstats"])
        assert obs["chars"][blstats.y, blstats.x] == ord("@")

        return obs

    def fork(self):
        fork_again = True
        while fork_again:
            pid = fork_with_nethack_env(self.env)
            if pid != 0:
                # parent
                print("freezing parent")
                while 1:
                    try:
                        os.waitpid(pid, 0)
                        break
                    except KeyboardInterrupt:
                        pass
                self.visualizer.force_next_frame()
                self.visualizer.render()
                while 1:
                    try:
                        fork_again = input("fork again [yn]: ")
                        if fork_again == "y":
                            fork_again = True
                            break
                        elif fork_again == "n":
                            fork_again = False
                            break
                    except KeyboardInterrupt:
                        pass

                termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            else:
                # child
                atexit.unregister(multiprocessing.util._exit_function)
                self.visualizer.force_next_frame()
                self.visualizer.render()
                break

    def render(self, force=False):
        if self.visualizer is not None:
            with self.debug_tiles(
                self.agent.current_level().walkable, color=(0, 255, 0, 128)
            ) if self.draw_walkable else contextlib.suppress():
                with self.debug_tiles(
                    ~self.agent.current_level().seen, color=(255, 0, 0, 128)
                ) if self.draw_seen else contextlib.suppress():
                    with self.debug_tiles(
                        self.agent.current_level().shop, color=(0, 0, 255, 64)
                    ) if self.draw_shop else contextlib.suppress():
                        with self.debug_tiles(
                            self.agent.current_level().shop_interior,
                            color=(0, 0, 255, 64),
                        ) if self.draw_shop else contextlib.suppress():
                            with self.debug_tiles(
                                (self.last_observation["specials"] & nh.MG_OBJPILE) > 0,
                                color=(0, 255, 255, 128),
                            ):
                                with self.debug_tiles(
                                    [self.agent.cursor_pos], color=(255, 255, 255, 128)
                                ):
                                    if force:
                                        self.visualizer.force_next_frame()
                                    rendered = self.visualizer.render()

            if not force and (not self.interactive or not rendered):
                return

            if self.agent is not None:
                print("Message:", self.agent.message)
                print("Pop-up :", self.agent.popup)
            print()
            if self.agent is not None and hasattr(self.agent, "blstats"):
                print(agent_lib.BLStats(*self.last_observation["blstats"]))
                print(
                    f"Carrying: {self.agent.inventory.items.total_weight} / {self.agent.character.carrying_capacity}"
                )
                print("Character:", self.agent.character)
            print("Misc :", self.last_observation["misc"])
            print("Score:", self.score)
            print("Steps:", self.env._steps)
            print("Turns:", self.env._turns)
            # print("Seed :", self.env.get_seeds())
            print("Items below me :", self.agent.inventory.items_below_me)
            print("Engraving below me:", self.agent.inventory.engraving_below_me)
            print()
            print(self.agent.inventory.items)
            print("-" * 20)

            self.env.render()
            print("-" * 20)
            print()

    def print_help(self):
        scene_glyphs = set(self.env.last_observation[0].reshape(-1))
        obj_classes = {getattr(nh, x): x for x in dir(nh) if x.endswith("_CLASS")}
        glyph_classes = sorted(
            (getattr(nh, x), x) for x in dir(nh) if x.endswith("_OFF")
        )

        texts = []
        for i in range(nh.MAX_GLYPH):
            desc = ""
            if glyph_classes and i == glyph_classes[0][0]:
                cls = glyph_classes.pop(0)[1]

            if nh.glyph_is_monster(i):
                desc = f': "{nh.permonst(nh.glyph_to_mon(i)).mname}"'

            if nh.glyph_is_normal_object(i):
                obj = nh.objclass(nh.glyph_to_obj(i))
                appearance = nh.OBJ_DESCR(obj) or nh.OBJ_NAME(obj)
                oclass = ord(obj.oc_class)
                desc = f': {obj_classes[oclass]}: "{appearance}"'

            desc2 = "Labels: "
            if i in agent_lib.G.INV_DICT:
                desc2 += ",".join(agent_lib.G.INV_DICT[i])

            if i in scene_glyphs:
                pos = (self.env.last_observation[0].reshape(-1) == i).nonzero()[0]
                count = len(pos)
                pos = pos[0]
                char = bytes([self.env.last_observation[1].reshape(-1)[pos]])
                texts.append(
                    (
                        -count,
                        f'{" " if i in agent_lib.G.INV_DICT else "U"} Glyph {i:4d} -> '
                        f"Char: {char} Count: {count:4d} "
                        f'Type: {cls.replace("_OFF", ""):11s} {desc:30s} '
                        f'{agent_lib.ALL.find(i) if agent_lib.ALL.find(i) is not None else "":20} '
                        f"{desc2}",
                    )
                )
        for _, t in sorted(texts):
            print(t)

    def get_action(self):
        while 1:
            key = os.read(sys.stdin.fileno(), 5)

            if key == b"\x1bOP":  # F1
                self.draw_walkable = not self.draw_walkable
                self.visualizer.force_next_frame()
                self.render()
                continue
            elif key == b"\x1bOQ":  # F2
                self.draw_seen = not self.draw_seen
                self.visualizer.force_next_frame()
                self.render()
                continue

            elif key == b"\x1bOR":  # F3
                self.draw_shop = not self.draw_shop
                self.visualizer.force_next_frame()
                self.render()
                continue

            if key == b"\x1bOS":  # F4
                raise ReloadAgent()

            if key == b"\x1b[15~":  # F5
                self.fork()
                continue

            elif key == b"\x1b[3~":  # Delete
                self.to_skip = 16
                return None

            if len(key) != 1:
                print("wrong key", key)
                continue
            key = key[0]
            if key == 10:
                key = 13

            if key == 63:  # '?"
                self.print_help()
                continue
            elif key == 127:  # Backspace
                self.visualizer.force_next_frame()
                return None
            else:
                actions = [a for a in self.env.actions if int(a) == key]
                assert len(actions) < 2
                if len(actions) == 0:
                    print("wrong key", key)
                    continue

                action = actions[0]
                return action

    def step(self, agent_action):
        if self.visualizer is not None and self.visualizer.video_writer is None:
            self.visualizer.step(self.last_observation, repr(chr(int(agent_action))))

            if self.interactive and self.to_skip <= 1:
                self.visualizer.force_next_frame()
            self.render()

            if self.interactive:
                print()
                print("agent_action:", agent_action, repr(chr(int(agent_action))))
                print()

            if self.to_skip > 0:
                self.to_skip -= 1
                action = None
            else:
                action = self.get_action()

            if action is None:
                action = agent_action

            if self.interactive:
                print("action:", action)
                print()
        else:
            if self.visualizer is not None:
                self.visualizer.step(
                    self.last_observation, repr(chr(int(agent_action)))
                )
            action = agent_action

        obs, reward, done, info = self.env.step(self.env.actions.index(action))
        self.score += reward
        self.step_count += 1
        # if not done:
        #     agent_lib.G.assert_map(obs['glyphs'], obs['chars'])

        # uncomment to debug measure up to assumed median
        # if self.score >= 7000:
        #     done = True
        #     self.end_reason = 'quit after median'
        if done:
            if self.visualizer is not None:
                self.visualizer.step(
                    self.last_observation, repr(chr(int(agent_action)))
                )

            end_reason = (
                bytes(obs["tty_chars"].reshape(-1))
                .decode()
                .replace("You made the top ten list!", "")
                .split()
            )
            if end_reason[7].startswith("Agent"):
                self.score = int(end_reason[6])
                end_reason = " ".join(end_reason[8:-2])
            else:
                assert self.score == 0, end_reason
                end_reason = " ".join(end_reason[7:-2])
            first_sentence = end_reason.split(".")[0].split()
            self.end_reason = (
                info["end_status"].name
                + ": "
                + (
                    " ".join(first_sentence[: first_sentence.index("in")])
                    + ". "
                    + ".".join(end_reason.split(".")[1:]).strip()
                ).strip()
            )
        if self.step_limit is not None and self.step_count == self.step_limit + 1:
            self.end_reason = self.end_reason or "steplimit"
            done = True
        elif self.step_limit is not None and self.step_count > self.step_limit + 1:
            assert 0

        self.last_observation = obs

        if done:
            self.is_done = True
            if self.visualizer is not None:
                self.render()
            if self.interactive:
                print("Summary:")
                pprint(self.get_summary())

        return obs, reward, done, info

    def debug_tiles(self, *args, **kwargs):
        if self.visualizer is not None:
            return self.visualizer.debug_tiles(*args, **kwargs)
        return contextlib.suppress()

    def debug_log(self, txt, color=(255, 255, 255)):
        if self.visualizer is not None:
            return self.visualizer.debug_log(txt, color)
        return contextlib.suppress()

    def get_data(self):
        return self.agent._data

    def get_summary(self):
        return {
            "score": self.score,
            "steps": self.env._steps,
            "turns": self.agent.blstats.time,
            "level_num": len(self.agent.levels),
            "experience_level": self.agent.blstats.experience_level,
            "milestone": self.agent.global_logic.milestone,
            "panic_num": len(self.agent.all_panics),
            "character": str(self.agent.character).split()[0],
            "end_reason": self.end_reason,
            # "seed": self.env.get_seeds(),
            **self.agent.stats_logger.get_stats_dict(),
        }
