import os
import jsonlines
import multiprocessing
from argparse import ArgumentParser
from tqdm import tqdm
import gym
from nle.nethack.actions import ACTIONS
from autoascend.env_wrapper import EnvWrapper
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv
from prompt_builder import ConcatPromptBuilder, DiffPromptBuilder

nle_action_textmap = {
    "UnsafeActions.HELP": "help",
    "UnsafeActions.PREVMSG": "previous message",
    "CompassDirection.N": "north",
    "CompassDirection.E": "east",
    "CompassDirection.S": "south",
    "CompassDirection.W": "west",
    "MiscDirection.UP": "up",
    "MiscDirection.DOWN": "down",
    "MiscDirection.WAIT": "wait",
    "MiscAction.MORE": "more",
    "Command.EXTCMD": "extcmd",
    "Command.EXTLIST": "extlist",
    "Command.ADJUST": "adjust",
    "Command.ANNOTATE": "annotate",
    "Command.APPLY": "apply",
    "Command.ATTRIBUTES": "attributes",
    "Command.AUTOPICKUP": "autopickup",
    "Command.CALL": "call",
    "Command.CAST": "cast",
    "Command.CHAT": "chat",
    "Command.CLOSE": "close",
    "Command.CONDUCT": "conduct",
    "Command.DIP": "dip",
    "Command.DROP": "drop",
    "Command.DROPTYPE": "droptype",
    "Command.EAT": "eat",
    "Command.ESC": "esc",
    "Command.ENGRAVE": "engrave",
    "Command.ENHANCE": "enhance",
    "Command.FIRE": "fire",
    "Command.FIGHT": "fight",
    "Command.FORCE": "force",
    "Command.GLANCE": "glance",
    "Command.HISTORY": "history",
    "Command.INVENTORY": "inventory",
    "Command.INVENTTYPE": "inventtype",
    "Command.INVOKE": "invoke",
    "Command.JUMP": "jump",
    "Command.KICK": "kick",
    "Command.KNOWN": "known",
    "Command.KNOWNCLASS": "knownclass",
    "Command.LOOK": "look",
    "Command.LOOT": "loot",
    "Command.MONSTER": "monster",
    "Command.MOVE": "move",
    "Command.MOVEFAR": "movefar",
    "Command.OFFER": "offer",
    "Command.OPEN": "open",
    "Command.OPTIONS": "options",
    "Command.OVERVIEW": "wizard where",
    "Command.PAY": "pay",
    "Command.PICKUP": "pickup",
    "Command.PRAY": "pray",
    "Command.PUTON": "puton",
    "Command.QUAFF": "quaff",
    "Command.QUIT": "quit",
    "Command.QUIVER": "quiver",
    "Command.READ": "read",
    "Command.REDRAW": "redraw",
    "Command.REMOVE": "remove",
    "Command.RIDE": "ride",
    "Command.RUB": "rub",
    "Command.RUSH": "rush",
    "Command.RUSH2": "rush2",
    "Command.SAVE": "save",
    "Command.SEARCH": "search",
    "Command.SEEALL": "seeall",
    "Command.SEEAMULET": "seeamulet",
    "Command.SEEARMOR": "seearmor",
    "Command.SEEGOLD": "seegold",
    "Command.SEERINGS": "seerings",
    "Command.SEESPELLS": "seespells",
    "Command.SEETOOLS": "seetools",
    "Command.SEETRAP": "seetrap",
    "Command.SEEWEAPON": "seeweapon",
    "Command.SHELL": "shell",
    "Command.SIT": "sit",
    "Command.SWAP": "swap",
    "Command.TAKEOFF": "takeoff",
    "Command.TAKEOFFALL": "takeoffall",
    "Command.TELEPORT": "teleport",
    "Command.THROW": "throw",
    "Command.TIP": "tip",
    "Command.TRAVEL": "travel",
    "Command.TURN": "turnundead",
    "Command.TWOWEAPON": "twoweapon",
    "Command.UNTRAP": "untrap",
    "Command.VERSION": "version",
    "Command.VERSIONSHORT": "versionshort",
    "Command.WEAR": "wear",
    "Command.WHATDOES": "whatdoes",
    "Command.WHATIS": "whatis",
    "Command.WIELD": "wield",
    "Command.WIPE": "wipe",
    "Command.ZAP": "zap",
    "TextCharacters.MINUS": "minus",
    "TextCharacters.SPACE": "space",
    "TextCharacters.APOS": "apos",
    "TextCharacters.NUM_0": "zero",
    "TextCharacters.NUM_1": "one",
    "TextCharacters.NUM_2": "two",
    "TextCharacters.NUM_3": "three",
    "TextCharacters.NUM_4": "four",
    "TextCharacters.NUM_5": "five",
    "TextCharacters.NUM_6": "six",
    "TextCharacters.NUM_7": "seven",
    "TextCharacters.NUM_8": "eight",
    "TextCharacters.NUM_9": "nine",
    "WizardCommand.WIZDETECT": "wizard detect",
    "WizardCommand.WIZGENESIS": "wizard genesis",
    "WizardCommand.WIZIDENTIFY": "wizard identify",
    "WizardCommand.WIZLEVELPORT": "wizard teleport",
    "WizardCommand.WIZMAP": "wizard map",
    "WizardCommand.WIZWISH": "wizard wish",
    "CompassDirection.NE": "northeast",
    "CompassDirection.SE": "southeast",
    "CompassDirection.SW": "southwest",
    "CompassDirection.NW": "northwest",
    "CompassDirectionLonger.N": "far north",
    "CompassDirectionLonger.E": "far east",
    "CompassDirectionLonger.S": "far south",
    "CompassDirectionLonger.W": "far west",
    "CompassDirectionLonger.NE": "far northeast",
    "CompassDirectionLonger.SE": "far southeast",
    "CompassDirectionLonger.SW": "far southwest",
    "CompassDirectionLonger.NW": "far northwest",
    "TextCharacters.PLUS": "+",
    "TextCharacters.QUOTE": '"',
    "TextCharacters.DOLLAR": "$",
}


NH_ACTION_STR_TO_IDX = {str(ACTIONS[i]): i for i in range(len(ACTIONS))}
# NH_ACTION_IDX_TO_STR = {v: k for (k, v) in NH_ACTION_STR_TO_IDX.items()}


# Inefficient. Ideally we want to use the C code to render the tty_chars directly
def ascii_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        result += "\n"
        for j in range(cols):
            entry = chr(chars[i, j])
            result += entry
    return result


def gen_and_write_episode(idx, start_idx, total_rollouts, data_dir, max_length):
    nle_language = NLELanguageObsv()

    with tqdm(total=total_rollouts, position=idx) as pbar:
        for game_id in range(start_idx, start_idx + total_rollouts):
            env = EnvWrapper(
                gym.make("NetHackChallenge-v0", no_progress_timeout=100),
                agent_args=dict(panic_on_errors=True, verbose=False),
                step_limit=10000000000,
            )

            try:
                env.main()
            except BaseException:
                pass

            # # Don't really need this either
            # summary = env.get_summary()
            # json_safe_summary = {}
            # for key, val in summary.items():
            #     if (
            #         isinstance(val, int)
            #         or isinstance(val, str)
            #         or isinstance(val, float)
            #         or isinstance(val, tuple)
            #     ):
            #         json_safe_summary[key] = val
            #     else:
            #         json_safe_summary[key] = val.item()
            # text_data = [json_safe_summary]

            data = env.get_data()

            datarows = []
            prompt_builder = ConcatPromptBuilder(
                max_length=max_length,
                prefix="You are an agent playing NetHack. Predict the next keypresses.\n\n",
            )
            for ts in range(len(data)):
                datum = data[ts]

                ascii_map = ascii_render(datum["tty_chars"])

                hybrid_obsv = f"""\nInventory:\n{nle_language.text_inventory(
                    datum["inv_strs"], datum["inv_letters"]
                ).decode("latin-1")}\nMap observation:\n{nle_language.text_glyphs(
                        datum["glyphs"], datum["blstats"]
                    ).decode("latin-1")}\n{ascii_map}\n"""

                # Previous observation
                # txt_blstats = nle_language.text_blstats(datum["blstats"]).decode(
                #     "latin-1"
                # )
                # txt_glyphs =
                # txt_message = nle_language.text_message(datum["tty_chars"]).decode(
                #     "latin-1"
                # )
                # txt_inventory = nle_language.text_inventory(
                #     datum["inv_strs"], datum["inv_letters"]
                # ).decode("latin-1")
                # txt_cursor = (
                #     nle_language.text_cursor(
                #         datum["glyphs"], datum["blstats"], datum["tty_chars"]
                #     ).decode("latin-1"),
                # )

                if ts < len(data) - 1:
                    txt_action = nle_action_textmap[data[ts + 1]["action"]]
                else:
                    txt_action = "esc"

                # text_obs = nle_text_obs(
                #     {
                #         "text_blstats": txt_blstats,
                #         "text_glyphs": txt_glyphs,
                #         "text_message": txt_message,
                #         "text_inventory": txt_inventory,
                #         "text_cursor": txt_cursor,
                #         # "text_action": txt_action,
                #     }
                # )
                prompt_builder.append_observation(hybrid_obsv)

                datarows.append(
                    {"prompt": prompt_builder.get_prompt(), "completion": txt_action}
                )

                # # Not doing this for now
                # if vision_version:
                #     vision_datum = {
                #         "tty_chars": datum["tty_chars"].tolist(),
                #         "tty_colors": datum["tty_colors"].tolist(),
                #         "tty_cursor": datum["tty_cursor"].tolist(),
                #     }
                #     if ts < len(data) - 1:
                #         action = NH_ACTION_STR_TO_IDX[data[ts + 1]["action"]]
                #     else:
                #         action = NH_ACTION_STR_TO_IDX["Command.ESC"]
                #     vision_datum["int_action"] = action
                #     text_datum = {**text_datum, **vision_datum}

                # text_data += [text_datum]

            fn = f"{game_id}_{len(data)}.jsonl"
            with jsonlines.open(os.path.join(data_dir, fn), "w") as writer:
                writer.write_all(datarows)

            pbar.update(1)

    return 1


def create_dataset(args, use_multiprocessing=True):
    # Configure data directory based on roles and episodes
    data_dir = os.path.join(args.base_dir, f"{args.episodes}")
    os.makedirs(data_dir, exist_ok=True)

    if use_multiprocessing:
        # Calculate number of processes and distribution of work using multiprocessing
        total_eps = args.episodes
        num_procs = min(multiprocessing.cpu_count() - args.cores_to_reserve, total_eps)
        episode_counts = [
            (
                (total_eps // num_procs) + 1
                if i < total_eps % num_procs
                else (total_eps // num_procs)
            )
            for i in range(num_procs)
        ]
        pool = multiprocessing.Pool(num_procs)
        tasks = []
        start_idx = 0
        for process_id, episode_count in enumerate(episode_counts):
            tasks.append(
                pool.apply_async(
                    gen_and_write_episode,
                    (process_id, start_idx, episode_count, data_dir, args.max_length),
                )
            )
            start_idx += episode_count

        # Wait for all tasks to complete
        results = [task.get() for task in tasks]
        pool.close()
        pool.join()
    else:
        # Sequential processing
        gen_and_write_episode(0, 0, args.episodes, data_dir, args.max_length)

    print("Dataset generation complete.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base_dir", default="data", type=str, help="dir where to store data"
    )
    parser.add_argument("--vision_version", default=0, type=int)  # currently unused
    parser.add_argument("-n", "--episodes", type=int, default=10)
    parser.add_argument(
        "--panic-on-errors", default=True, action="store_true"
    )  # what do?
    parser.add_argument("--cores_to_reserve", type=int, default=0)  # what do?
    parser.add_argument("--max_length", type=int, default=8000)  # what do?
    args = parser.parse_args()

    print("ARGS:", args)
    return args


def main():
    args = parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
