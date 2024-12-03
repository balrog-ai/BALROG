# Few Shot Learning
The approach enhances model performance by providing examples of expert gameplay in the agent's context.

### Installation
Download and unzip our expert demonstrations
```bash
pip install gdown
gdown 1TQbrqMSC5K_SNx9tta1Tlhtg8flSIGaJ
unzip records.zip
```

### Usage
To run Few-Shot Learning 
```bash
python -m eval agent.type=few_shot eval.icl_episodes=5
```

### Features
- each demonstration have corresponding mp4 file, which allows for quick inspection
- `FewShotAgent` allows for context caching, can be enabled with `agent.cache_icl=True`

### Additional Notes:
- Expert demonstrations are formatted as conversation sequences
- all trajectories are loaded in context, this can increase the cost of evaluation, especially for environments like nethack
- for textworld environments we avoid the case where we put the solution into the context
- in principle we also could incorporate similar strategy for other environments, for example in nle we could load trajectories corresponding to the same character

### Prompt formatting
Example prompt for the agent starting playing the game with `eval.icl_episodes=1`
```
00 = Message(role=user, content=System Prompt: [], attachment=None)
01 = Message(role=user, content=****** START OF DEMONSTRATION EPISODE 1 ******, attachment=None)
02 = Message(role=user, content=Obesrvation: [], attachment=None)
03 = Message(role=assistant, content=None, attachment=None)
04 = Message(role=user, content=Obesrvation: [], attachment=None)
05 = Message(role=assistant, content=go forward, attachment=None)
06 = Message(role=user, content=Obesrvation: [], attachment=None)
07 = Message(role=assistant, content=go forward, attachment=None)
08 = Message(role=user, content=Obesrvation: [], attachment=None)
09 = Message(role=assistant, content=turn left, attachment=None)
10 = Message(role=user, content=****** END OF DEMONSTRATION EPISODE 1 ******, attachment=None)
11 = Message(role=user, content=****** Now it's your turn to play the game! ******, attachment=None)
12 = Message(role=user, content=Current Observation: [], attachment=None)
```
