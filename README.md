# NetHack benchmark for LLM agents

# Observation
```
Inventory:
a: a +0 katana (weapon in hand)
b: a +0 wakizashi (alternate weapon; not wielded)
c: a +0 yumi
d: 43 +0 ya (in quiver)
e: a blessed rustproof +0 splint mail (being worn)

Map observation:
ebony wand very near eastnortheast
tame little dog very near west



                                                                                
                                                                                
                                                                                
                                                           --------------       
                                                           |..../.......|       
                                                          #-d.@.........|       
                                                          #|............|       
                                                -----     #|............|       
                                            ####....|   ###|............|       
                                            #   |...|   #  --------------       
                                            #   |.^.| ###                       
                               ------------ #   |...-##                         
                               .<.........| #   |...|                           
                               |..........| #   -----                           
                               |..........| #                                   
                               |..........| #                                   
                               |..........|##                                   
                               ...........-#                                    
                               ------------                                     
                                                                                
                                                                                
                                                                                
Agent the Hatamoto             St:14 Dx:18 Co:16 In:9 Wi:9 Ch:7 Lawful S:0      
Dlvl:1 $:0 HP:15(15) Pw:2(2) AC:4 Xp:1/0 T:72
```

# Installation
NLE requires python>=3.5, cmake>=3.15 to be installed and available both when building the package, and at runtime.

On MacOS, one can use Homebrew as follows:
```
brew install cmake
```

On a plain Ubuntu 18.04 distribution, cmake and other dependencies can be installed by doing:
```
sudo apt-get install -y build-essential autoconf libtool pkg-config \
    python3-dev python3-pip python3-numpy git flex bison libbz2-dev

# recent cmake version
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update && apt-get --allow-unauthenticated install -y \
    cmake \
    kitware-archive-keyring
```

Afterwards it's a matter of setting up your environment. We advise using a conda environment for this:

```
conda create -y -n nle python=3.10
conda activate nle
pip install nle
```

# Install modified nle-language-wrapper
```
cd external/nle-language-wrapper
pip install -e ".[dev]"
```

## Download dataset and setup dataset
Instructions to download human data here: https://github.com/facebookresearch/nle/blob/main/DATASET.md

Make sure you have a `nld-nao/nld-nao-unzipped` dataset, and unzip all the files including the xlog files and blacklist in it. 

# RUN

### 1 - Run IDM to Label human games
`python label_human_games.py`

### 2 - Generate dataset .csv
`python generate_human_dataset.py`

### 3 - Run finetuning
`python finetune.py`


## Options:
Modify or create new the config files to change experiment types. Use config files to keep track of experiments.