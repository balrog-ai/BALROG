# Boxoban episode budget

The original MiniHack budget of 100 actions can truncate a solvable Boxoban
instance before its shortest solution finishes. With BALROG `b7afe79e`,
MiniHack `3ecb6da4`, Boxoban levels `78011d8c` and `balrog-nle==0.9.0`,
`MiniHack-Boxoban-Hard-v0` at seed 0 has a shortest four-direction solution
of 104 actions. Seed Python and NumPy immediately before `env.reset(seed=0)`,
as the original evaluator does; seeding only before construction selects a
different Boxoban map.

An independent breadth-first search over exact player and boulder positions
exhausted all 796,766 states reachable within 100 actions without a goal.
Its first goal occurs at depth 104, without heuristic or deadlock pruning.
The included snapshot makes this search independent of map download order.

The native task exposes only north/east/south/west, and completion requires
all four boulders on fountains. A blocked-boulder squeeze is not available
to this instance's elf priest: two spellbooks and four water potions alone
weigh 180, whereas even maximum carrying capacity permits at most 150 for
that mechanic. The movement-only interface cannot drop or consume them.
There are no monsters or polymorph traps enabling a different form.

Two fresh native executions truncated at action 100 with progression 0.
In a separate experiment changing only the budget to 200, the same action
sequence completed at action 104 with progression 1 in both runs.

## Proposed configuration

`envs.minihack_kwargs.boxoban_max_episode_steps` overrides the general
MiniHack budget only for task IDs beginning with `MiniHack-Boxoban-`.
The proposed default is 200; all other MiniHack tasks retain 100.
Existing custom configurations without this key retain their old behavior.
To reproduce the previous uniform budget:

```bash
python eval.py envs.minihack_kwargs.boxoban_max_episode_steps=null
```

This changes the Boxoban evaluation budget and can change benchmark scores.
Record the setting with results and use the same budget on both sides of
any adapter parity experiment. A budget of 200 resolves the documented
instance; it does not guarantee solvability of every Boxoban map within 200.

## Reproduction

The standalone search needs only Python's standard library:

```bash
python docs/envs/boxoban_budget_repro.py
```

With BALROG and the pinned native dependencies/assets installed, additionally
check the real environment at both budgets:

```bash
python docs/envs/boxoban_budget_repro.py --native
```

Native reproduction checks the initial board against the snapshot and fails
if a different map is selected. Changing the source or asset installation
can change seed-to-map selection; it should not be hidden by replacing seeds.
