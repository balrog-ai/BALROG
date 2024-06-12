import os
import jsonlines
import multiprocessing
from argparse import ArgumentParser
from tqdm import tqdm
import gym
from nle.nethack.actions import ACTIONS
from autoascend.env_wrapper import EnvWrapper
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv
from fmrl.prompt_builder import ConcatHistoryPromptBuilder as ConcatPromptBuilder, DiffHistoryPromptBuilder as DiffPromptBuilder
import pickle

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

def gen_and_write_episode(game_id, data_dir, score_threshold):
    env = EnvWrapper(
        gym.make("NetHackChallenge-v0", no_progress_timeout=100),
        agent_args=dict(panic_on_errors=True, verbose=False),
        step_limit=10000000000,
    )

    try:
        env.main()
    except BaseException:
        pass

    summary = env.get_summary()
    
    # low quality, try again
    if summary["score"] < score_threshold: 
        return gen_and_write_episode(game_id, data_dir, score_threshold)

    data = env.get_data()
    data = [data[t] for t in range(len(data))]
        
    with open(os.path.join(data_dir, f"{game_id}_summary.pkl"), "wb") as f:
        pickle.dump(summary, f)
        
    with open(os.path.join(data_dir, f"{game_id}_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    # success!
    return 1
    
def worker(job_queue, result_queue, data_dir, score_threshold):
    while True:
        game_id = job_queue.get()
        if game_id is None:
            break
        result = gen_and_write_episode(game_id, data_dir, score_threshold)
        result_queue.put(result)

def generate_episodes(args, use_multiprocessing=True):
    data_dir = args.base_dir
    os.makedirs(data_dir, exist_ok=True)
    if use_multiprocessing:
        num_procs = min(multiprocessing.cpu_count() - args.cores_to_reserve, args.episodes)

        job_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        # fill job queue
        for game_id in range(args.episodes):
            job_queue.put(game_id)

        # launch processes
        print(f"Launching {num_procs} processes")
        processes = []
        for _ in range(num_procs):
            p = multiprocessing.Process(target=worker, args=(job_queue, result_queue, data_dir, args.score_threshold))
            p.start()
            processes.append(p)
            
        # track completed episodes
        results = []
        with tqdm(total=args.episodes) as pbar:
            while len(results) < args.episodes:
                results.append(result_queue.get())
                pbar.update(1)

        # stop all processes
        for p in processes:
            p.join()

        return results
    else:
        for game_id in tqdm(range(args.episodes)):
            gen_and_write_episode(game_id, data_dir, args.score_threshold)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--base_dir", default="data", type=str)
    parser.add_argument("-n", "--episodes", type=int, default=1000)
    parser.add_argument("--panic-on-errors", default=True, action="store_true") # what do?
    parser.add_argument("--cores_to_reserve", type=int, default=0) # what do?
    # parser.add_argument("--max_length", type=int, default=8000) # what do?
    parser.add_argument("--score_threshold", type=int, default=10000) # what do?
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    generate_episodes(parse_args())
