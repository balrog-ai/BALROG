from .env import CrafterLanguageWrapper, ACTIONS

def get_instruction_prompt(task=None):
    action_strings = ",\n".join(f"{action}" for action in ACTIONS)
    instruction_prompt = f"""
You are an agent playing Crafter. The following are the possible actions you can take in the game:

{action_strings}.

Write all information helpful for the game in a numbered list.
1. Collect resources such as wood, stone, and iron to craft tools and weapons.
2. Build shelters to protect yourself from monsters at night.
3. Use tools and weapons to defend yourself against monsters.
4. Build bridges to cross lakes and rivers.
5. Dig tunnels to surprise monsters and outsmart them.
6. Plant saplings and defend them against monsters to ensure a steady food supply.
7. Eat Cow to restore health.
8. Collect Drink to restore thirst.
9. Place a Plant to eat for health.
10. Make a Wood Pickaxe to collect Stone.
11. Make a Wood Sword to defeat Zombies.
12. Make a Stone Pickaxe to collect Iron.
13. Make a Stone Sword to defeat Skeletons.
14. Place a Furnace to smelt Iron.
15. Collect Coal to smelt Iron.
16. Collect Iron to make an Iron Pickaxe and Sword.
17. Make an Iron Pickaxe to collect Diamond.
18. Make an Iron Sword to defeat Zombies and Skeletons.
19. Collect Diamond to progress further.
20. Unlock achievements to receive rewards.
21. Wake Up to start the episode.

In plain text. List all objects I need to interact/avoid to survive in the game. Use "I would like to X object Y" in each step. Replace Y by the actual object, X by the actual interaction.
I would like to avoid zombies, skeletons, and spiders.
I would like to collect saplings.
I would like to craft a wood pickaxe.
I would like to collect wood.
I would like to craft a stone pickaxe.
I would like to collect stone.
I would like to craft a furnace.
I would like to collect coal.
I would like to collect iron.
I would like to craft an iron pickaxe.
I would like to collect diamonds.
I would like to craft an iron sword.
I would like to chase cows.
I would like to grow fruits.
I would like to drink from a lake.
I would like to sleep in a safe place.
I would like to craft a table.
I would like to eat food.
I would like to drink water.
I would like to rest.
I would like to build stone tools to defend myself against monsters.
I would like to build bridges to cross lakes.
I would like to dig tunnels to hide from monsters.
I would like to block arrows with stones.
I would like to dig through walls to surprise skeletons.
I would like to seek shelter in caves.
I would like to build plantations of saplings and defend them against monsters.
I would like to eat the growing fruits to ensure a steady food supply.
I would like to place a table.
I would like to eat a cow.
I would like to place a plant.
I would like to defeat a zombie.
I would like to place stone.
I would like to eat a plant.
I would like to defeat a skeleton.
I would like to wake up.
I would like to place a furnace.

Write all game objectives numbered list. For each objective, list its requirements.
1. Collect Wood: No requirements
2. Place Table: Requires Collect Wood
3. Eat Cow: No requirements
4. Collect Sampling: No requirements
5. Collect Drink: No requirements
6. Make Wood Pickaxe: Requires Place Table
7. Make Wood Sword: Requires Place Table
8. Place Plant: Requires Collect Sampling
9. Defeat Zombie: No requirements
10. Collect Stone: Requires Make Wood Pickaxe
11. Place Stone: Requires Collect Stone
12. Eat Plant: Requires Place Plant
13. Defeat Skeleton: No requirements
14. Make Stone Pickaxe: Requires Collect Stone
15. Make Stone Sword: Requires Collect Stone
16. Wake Up: No requirements
17. Place Furnace: Requires Collect Stone
18. Collect Coal: Requires Make Wood Pickaxe
19. Collect Iron: Requires Make Stone Pickaxe
20. Make Iron Pickaxe: Requires Place Furnace, Collect Coal, and Collect Iron
21. Make Iron Sword: Requires Place Furnace, Collect Coal, and Collect Iron
22. Collect Diamond: Requires Make Iron Pickaxe

Write all actions as a numbered list. For each action, list its requirements.
1. Move West: Flat ground left to the agent.
2. Move East: Flat ground right to the agent.
3. Move North: Flat ground above the agent.
4. Move South: Flat ground below the agent.
5. Do: Facing creature or material; have necessary tool.
6. Sleep: Energy level is below maximum.
7. Place Stone: Stone in inventory.
8. Place Table: Wood in inventory.
9. Place Furnace: Stone in inventory.
10. Place Plant: Sapling in inventory.
11. Make Wood Pickaxe: Nearby table; wood in inventory.
12. Make Stone Pickaxe: Nearby table; wood, stone in inventory.
13. Make Iron Pickaxe: Nearby table, furnace; wood, coal, iron an inventory.
14. Make Wood Sword: Nearby table; wood in inventory.
15. Make Stone Sword: Nearby table; wood, stone in inventory.
16. Make Iron Sword: Nearby table, furnace; wood, coal, iron in inventory.
17. Noop: Always applicable.

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible in the game.

PLAY!
""".strip()

    return instruction_prompt