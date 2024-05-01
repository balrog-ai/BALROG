import os
from tqdm import tqdm

import torch

import nle.dataset as nld
from nle.nethack import tty_render
from nle.nethack.actions import action_id_to_type
from nle.env.tasks import NetHackChallenge

# First download the NLD AA "Taster" dataset from https://dl.fbaipublicfiles.com/nld/nld-aa-taster/nld-aa-taster.zip
# and unzip it in the current directory.

# 1. Get the paths for your unzipped datasets
path_to_nld_aa_taster = "./nld-aa-taster/nle_data"

# 2. Chose a database name/path. By default, most methods with use nld.db.DB (='ttyrecs.db')
dbfilename = "ttyrecs_aa.db"

if not nld.db.exists(dbfilename):
    # 3. Create the db and add the directory
    nld.db.create(dbfilename)
    nld.add_nledata_directory(path_to_nld_aa_taster, "taster-dataset", dbfilename)

# Create a connection to specify the database to use
db_conn = nld.db.connect(filename=dbfilename)

# Then you can inspect the number of games in each dataset:
print(
    f"NLD AA \"Taster\" Dataset has {nld.db.count_games('taster-dataset', conn=db_conn)} games."
)


dataset = nld.TtyrecDataset(
    "taster-dataset",
    dbfilename=dbfilename,
)

# Create a list of gameids to use
num_games = nld.db.count_games("taster-dataset", conn=db_conn)
gameids = list(range(1, num_games + 1))


def ascii_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        result += "\n"
        for j in range(cols):
            entry = chr(chars[i, j])
            result += entry
    return result


for gameid in gameids:

    dataset = nld.TtyrecDataset(
        "taster-dataset",
        batch_size=1,
        seq_length=10000,
        dbfilename=dbfilename,
        gameids=[gameid],
    )

    print(f"The GAMEID is {gameid}")

    for idx, mb in enumerate(dataset):
        print(mb.keys())

        for i in range(len(mb["done"][0]) - 1):
            chars = mb["tty_chars"][0, i]
            print(ascii_render(chars))

            keypress = mb["keypresses"][0, i]
            action = action_id_to_type(keypress)

            print(action)

            if i > 100:
                break
        break
    break
