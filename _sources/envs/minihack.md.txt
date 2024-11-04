# MiniHack

MiniHack is a powerful sandbox framework built
on top of the NLE that enables researchers to
easily design rich and diverse environments for RL. It provides a
flexible platform for creating custom RL tasks ranging from simple
grid-world navigation to complex, procedurally generated worlds with
intricate game mechanics. The framework allows users to define
environments using a human-readable description language or a simple
Python interface, giving fine-grained control over environment elements
such as terrain, objects, monsters, and traps. MiniHack offers a diverse
array of tasks, which can be broadly categorized into three main groups:
Navigation Tasks, Skill Acquisition Tasks, and Ported Tasks. To enable
interaction with language models, we use the NetHack Language Wrapper.

From the MiniHack Navigation Tasks, we picked Corridor and
CorridorBattle, which challenge the agent to reach the goal position by
overcoming various difficulties on their way, such as fighting monsters
in corridors and navigating through complex or procedurally generated
mazes. These tasks feature a relatively small action space, i.e.,
movement towards 8 compass directions, and based on the environment,
search, kick, open, and eat actions.

![Examples of MiniHack Corridor
task.](../imgs/minihack_corridor.png)

![Example of MiniHack CorridorBattle
task.](../imgs/minihack_corridor_battle.png)

From the MiniHack Skill Acquisition Tasks, we picked Quest (with three
different difficulty levels, Easy, Medium, and Hard), which challenges
the agent to use objects found in the environment to cross a lava river
(these objects can provide levitation or freezing abilities), fight
monsters, navigate through rooms or mazes and towards the end of the
quest use a wand of death to defeat a powerful monster guarding the goal
location.

![Example of MiniHack Quest Hard
task.](../imgs/minihack_quest_hard.png)

We additionally test the agents on MiniHack Boxoban. This family of
environments is an adaptation of the Boxoban puzzle game, which itself
is inspired by the classic Sokoban. These environments present a
challenging puzzle-solving task within the MiniHack framework,
leveraging the NetHack game mechanics. The primary goal in MiniHack
Boxoban is to push four boulders (MiniHack's equivalent of boxes) onto
four designated goal locations, which are represented by fountains. This
task requires strategic thinking and planning, as the agent must
carefully maneuver the boulders through the environment without getting
them stuck in corners or against walls.

![Example of MiniHack Boxoban Hard
task.](../imgs/minihack_boxoban.png)

Standard errors were computed
using 5 seeds for each task. Here, GPT-4o and a Gemini-1.5-Pro equal
each other both in language-only and vision-language mode, with both
models only managing to complete some of the corridor and corridor
battle tasks. None of the other models solved any task.

## LLM results
| Model              |  Average Progress (%) |
| ------------------ | --------------------- |
| gpt-4o             |    5.71 &plusmn; 3.92    |
| gemini-1.5-pro     |    5.71 &plusmn; 3.92    |
| gpt-4o-mini        |    0.00 &plusmn; 0.00    |
| gemini-1.5-flash   |    0.00 &plusmn; 0.00    |
| llama-3.1-70B-it   |    0.00 &plusmn; 0.00    |
| llama-3.2-11B-it   |    0.00 &plusmn; 0.00    |
| llama-3.2-90B-it   |    0.00 &plusmn; 0.00    |


## VLM results
| Model              |  Average Progress (%)  |
| ------------------ | ---------------------- |
| gpt-4o             |    5.71 &plusmn; 3.92     |
| gemini-1.5-pro     |    5.71 &plusmn; 3.92     |
| gpt-4o-mini        |    0.00 &plusmn; 0.00     |
| gemini-1.5-flash   |    0.00 &plusmn; 0.00     |
| llama-3.2-11B-it   |    0.00 &plusmn; 0.00     |
| llama-3.2-90B-it   |    0.00 &plusmn; 0.00     |

## Observations

TODO
