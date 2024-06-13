import numpy as np
import difflib
from inverse_dynamics.idm import *
import nle.dataset as nld


def setup_dataset(path_to_nld_nao_data):
    # concatenate the path to the dbfilename
    dbfilename = path_to_nld_nao_data + "/nld-nao-unzipped" + "/ttyrecs_nao.db"

    print(dbfilename)

    if not nld.db.exists(dbfilename):
        # Create the db and add the directory
        print("Creating dataset")
        nld.db.create(dbfilename)
        nld.add_altorg_directory(
            path_to_nld_nao_data + "/nld-nao-unzipped", "nld-nao-dataset", dbfilename
        )

    # Create a connection to specify the database to use
    db_conn = nld.db.connect(filename=dbfilename)

    # Then you can inspect the number of games in each dataset:
    print(
        f"\"NLD NAO\" Dataset has {nld.db.count_games('nld-nao-dataset', conn=db_conn)} games."
    )

    return dbfilename


def main():

    base_path = "/Users/davidepaglieri/Desktop/repos/nle/nld-nao"
    dbfilename = setup_dataset(base_path)
    gameid = 145461  # Winning game, quite short too! 30K steps
    gameid = 2288666  # Very short winning game. Good player (winning in 22K steps)
    dataset = nld.TtyrecDataset(
        "nld-nao-dataset",
        batch_size=1,
        seq_length=1000,
        rows=24,
        cols=120,
        dbfilename=dbfilename,
        gameids=[gameid],
    )

    from copy import deepcopy

    tty_chars = []
    tty_cursors = []
    tty_colors = []

    finished = False

    for idx, mb in enumerate(dataset):
        if not (0 <= idx <= 30):
            continue
        if not finished:
            for i in range(len(mb["done"][0]) - 1):
                chars = mb["tty_chars"][0, i]
                tty_chars.append(deepcopy(mb["tty_chars"][0, i]))
                ttycursor = mb["tty_cursor"][0, i]
                ttycursor = np.array((ttycursor[1], ttycursor[0]))
                tty_cursors.append(deepcopy(ttycursor))
                tty_colors.append(deepcopy(mb["tty_colors"][0, i]))

                message = "".join([chr(c) for c in chars[0]])
                # print(message)
                if "You ascend t" in message:
                    print("ASCEND")
                    finished = True
                    break
                elif "Do you want your possessions identified?" in message:
                    print("DEAD")
                    finished = True
                    break
    np.savez(
        "game.npz", tty_chars=tty_chars, tty_cursor=tty_cursors, tty_colors=tty_colors
    )

    file = "game.npz"
    # game = np.load(file, allow_pickle=True)

    idm = IDM()
    output_file = idm.process_game(file)


if __name__ == "__main__":
    main()
