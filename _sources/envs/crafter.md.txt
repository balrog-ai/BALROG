# Crafter

Crafter is an open-source 2D survival game
designed specifically for research on strong generalization, deep
exploration, and long-term reasoning in reinforcement learning. It is a
Minecraft-inspired, procedurally generated environment that combines
resource gathering, crafting, and combat elements. Additionally, the
game includes a comprehensive set of tasks and achievements, enabling
researchers to evaluate agent performance across multiple objectives and
time scales. To enable interaction with language models we use the same
language wrapper as proposed in @wu2023smartplay.

![Crafter's examples of unique procedurally generated
maps.](../imgs/crafter_map.png)

## Crafter Results

Standard errors are computed using 10 seeds.
GPT4o leads in language-only mode, and Gemini-1.5-Pro leads in
vision-language mode. Surprisingly, Llama 3.2 90B performance decreases
very sharply when images are added, getting worse average progress than
its smaller 11B model.


## LLM results
| Model              |  Average Progress (%) |
| ------------------ | ----------------------|
| gpt-4o             |    33.10 &plusmn; 2.32   |
| llama-3.2-90B-it   |    31.69 &plusmn; 1.36   |
| llama-3.1-70B-it   |    31.31 &plusmn; 2.68   |
| gemini-1.5-pro     |    30.21 &plusmn; 2.86   |
| llama-3.2-11B-it   |    26.20 &plusmn; 3.30   |
| gemini-1.5-flash   |    20.00 &plusmn; 0.74   |
| gpt-4o-mini        |    12.72 &plusmn; 1.13   |


## VLM results
| Model              |  Average Progress (%)  |
| ------------------ | ---------------------- |
| gemini-1.5-pro     |    33.50 &plusmn; 2.07    |
| gpt-4o             |    26.81 &plusmn; 3.74    |
| llama-3.2-11B-it   |    23.63 &plusmn; 1.48    |
| gemini-1.5-flash   |    20.70 &plusmn; 4.43    |
| gpt-4o-mini        |    19.91 &plusmn; 3.13    |
| llama-3.2-90B-it   |    10.00 &plusmn; 1.13    |

## Observations

TODO
