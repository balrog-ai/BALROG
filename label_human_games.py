import os
import numpy as np
from idm.inverse_dynamics.idm import IDM
from copy import deepcopy
import nle.dataset as nld
from multiprocessing import Pool
import sys
import sqlite3
from omegaconf import OmegaConf

def setup_dataset(path_to_nld_nao_data):
    # concatenate the path to the dbfilename
    dbfilename = path_to_nld_nao_data + "/ttyrecs_nao.db"

    print(dbfilename)

    if not nld.db.exists(dbfilename):
        # Create the db and add the directory
        print("Creating dataset")
        nld.db.create(dbfilename)
        nld.add_altorg_directory(path_to_nld_nao_data, "nld-nao-dataset", dbfilename)

    # Create a connection to specify the database to use
    db_conn = nld.db.connect(filename=dbfilename)

    # Then you can inspect the number of games in each dataset:
    print(
        f"\"NLD NAO\" Dataset has {nld.db.count_games('nld-nao-dataset', conn=db_conn)} games."
    )

    return dbfilename


def process_gameid(args):
    gameid, dbfilename, dlvl_cutoff = args

    try:
        dataset = nld.TtyrecDataset(
            "nld-nao-dataset",
            batch_size=1,
            seq_length=1000,
            rows=24,
            cols=120,
            dbfilename=dbfilename,
            gameids=[gameid],
        )
        finished = False
        tty_chars = []
        tty_cursors = []
        tty_colors = []

        game_path = f"human_labeled/{gameid}.npz"

        if not os.path.exists("human_labeled"):
            os.makedirs("human_labeled")

        for idx, mb in enumerate(dataset):
            if not finished:
                for i in range(len(mb["done"][0]) - 1):
                    chars = mb["tty_chars"][0, i]
                    tty_chars.append(deepcopy(mb["tty_chars"][0, i]))
                    ttycursor = mb["tty_cursor"][0, i]
                    ttycursor = np.array((ttycursor[1], ttycursor[0]))
                    tty_cursors.append(deepcopy(ttycursor))
                    tty_colors.append(deepcopy(mb["tty_colors"][0, i]))

                    message = "".join([chr(c) for c in chars[0]])
                    if "You ascend t" in message:
                        print("ASCEND")
                        finished = True
                        break
                    elif "Do you want your possessions identified?" in message:
                        print("DEAD")
                        finished = True
                        break

        np.savez(
            game_path,
            tty_chars=tty_chars,
            tty_cursor=tty_cursors,
            tty_colors=tty_colors,
        )

        idm = IDM()
        actions, inventory, summary = idm.label_game(game_path, dlvl_cutoff=dlvl_cutoff)

        np.savez_compressed(
            game_path,
            tty_chars=tty_chars,
            tty_cursor=tty_cursors,
            tty_colors=tty_colors,
            action=actions,
            inventory=inventory,
            summary=summary,
        )
    except Exception as e:
        print(f"Error processing gameid {gameid}: {e}")



def main(config):
    base_path = "/root/nld-nao/nld-nao-unzipped"
    max_games = config.max_games

    dbfilename = setup_dataset(base_path)

    conn = sqlite3.connect(f"{base_path}/ttyrecs_nao.db")
    cursor = conn.cursor()

    query = config.query

    cursor.execute(query)
    gameids = cursor.fetchall()

    gameids = [gameid[0] for gameid in gameids]
    print(f"Games found before check: {len(gameids)}")

    gameids_processed = []

    # Read corrupted game list, and remove any of the gameids that are in that list
    with open("idm/corrupted_games.txt", "r") as file:
        corrupted_games = [int(gameid) for gameid in file.readlines()]

    gameids = [gameid for gameid in gameids if gameid not in corrupted_games]
    print(f"Non corrupted games: {len(gameids)}")
    gameids = gameids[:max_games]
    print(f"Games about to be labelled: {len(gameids)}")

    with Pool(processes=config.processes) as pool:
        pool.map(
            process_gameid,
            [(gameid, dbfilename, config.dlvl_cutoff) for gameid in gameids],
        )
        gameids_processed.extend(gameids)

    # Write the gameids processed to a file
    with open("human_labeled/gameids.txt", "w") as file:
        for gameid in gameids_processed:
            file.write(f"{gameid}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config/label_human_games.yaml"
    
    config = OmegaConf.load(config_file)
    
    main(config)
