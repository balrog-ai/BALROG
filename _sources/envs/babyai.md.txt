# Baby AI

BabyAI is a research platform designed to study
grounded language learning and instruction following in artificial
agents. It consists of a suite of 2D grid world environments with
increasing levels of complexity. In these environments, an agent
navigates through rooms and interacts with various objects like doors,
keys, balls, and boxes of different colors. The agent receives natural
language instructions, called "missions\", which describe tasks it needs
to complete, such as picking up specific objects or navigating to
certain locations. Many existing works on decision-making have studied
model performance on this environment.
We use it as a historically relevant environment that we expect to be
relatively easy to solve.

## BabyAI-Text

We evaluate the agents on 5 tasks introduced in
BabyAI-Text, which provides a description of each
observation instead of a symbolic representation. A textual description
consists of a list of template descriptions with the following
structure:

-   "You see a `<object>` `<location>`" if the object is a key, a ball,
    a box or a wall.

-   "You see a(n) open/closed door `<location>`" , if the agent sees a
    door.

-   "You carry a `<object>`", if the agent carries an object.

## BabyAI Results

We provide BabyAI results for LLM and VLM mode. Errors are computed with 25 seeds for each of
the 5 tasks of BabyAI. GPT-4o leads, closely followed by Llama 3.1 70B.
When vision is added to the observation, GPT4o all models performance
decrease, except for Gemini-1.5-Pro, whose performance remains stable.


### LLM results
  | Model                                               |  Average Progress (%)  | 
  | --------------------------------------------------- | ---------------------- | 
  | gpt-4o                                              |    77.60 &plusmn; 3.73    | 
  | llama-3.1-70B-it                                    |    73.20 &plusmn; 3.96    | 
  | gemini-1.5-pro                                      |    58.40 &plusmn; 4.41    | 
  | llama-3.2-90B-it                                    |    55.20 &plusmn; 4.45    | 
  | gpt-4o-mini                                         |    50.40 &plusmn; 4.47    | 
  | llama-3.2-11B-it                                    |    32.80 &plusmn; 4.20    | 
  | gemini-1.5-flash                                    |    25.60 &plusmn; 3.90    | 


### VLM results
  | Model                                               |  Average Progress (%)  |
  | --------------------------------------------------- | ---------------------- |
  | gpt-4o                                              |    62.00 &plusmn; 4.34    |
  | gemini-1.5-pro                                      |    58.40 &plusmn; 4.41    |
  | gemini-1.5-flash                                    |    43.20 &plusmn; 4.43    |
  | gpt-4o-mini                                         |    38.00 &plusmn; 4.34    |
  | llama-3.2-90B-it                                    |    28.20 &plusmn; 4.02    |
  | llama-3.2-11B-it                                    |    10.40 &plusmn; 2.73    |

Observations
------------

TODO
