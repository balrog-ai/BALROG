from typing import List, Optional

from balrog.agents.base import BaseAgent


class Message:
    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class FewShotRLAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.icl_episodes = []
        self.cached_icl = False

    def wrap_episode(self, events):
        icl_episode = []
        icl_episode.append(
            Message(role="user", content=f"****** START OF DEMONSTRATION EPISODE {len(self.icl_episodes) + 1} ******")
        )
        for event in events:
            if event["type"] == "observation":
                content = "Obesrvation:\n" + event["text"]
                message = Message(role="user", content=content)
            elif event["type"] == "action":
                content = event["action"]
                message = Message(role="assistant", content=content)
            icl_episode.append(message)
        icl_episode.append(
            Message(role="user", content=f"****** END OF DEMONSTRATION EPISODE {len(self.icl_episodes) + 1} ******")
        )

        self.icl_episodes.append(icl_episode)

    def get_icl_prompt(self) -> List[Message]:
        icl_instruction = Message(
            role="user",
            content=self.prompt_builder.system_prompt.replace(
                "PLAY",
                "First, observe the demonstrations provided and learn from them!",
            ),
        )

        # unroll the wrapped icl episodes messages
        icl_messages = [icl_instruction]
        for icl_episode in self.icl_episodes:
            icl_messages.extend(icl_episode)

        end_demo_message = Message(
            role="user",
            content="****** Now it's your turn to play the game! ******",
        )
        icl_messages.append(end_demo_message)

        return icl_messages

    def act(self, obs, info, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs, info)

        if "final_observation" in info:
            # add last episode to icl_episodes
            last_event = self.prompt_builder._events[-1]
            final_obs = last_event["info"]["final_observation"]
            final_info = last_event["info"]["final_info"]
            final_event = {
                "type": "observation",
                "text": final_obs["text"].get("long_term_context", ""),
                "image": final_obs.get("image", None),
                "info": final_info,
            }
            icl_events = list(self.prompt_builder._events)[:-1] + [final_event]

            # wrap the episode
            self.wrap_episode(icl_events)

            # remove last episode from probmpt builder events
            self.prompt_builder._events.clear()
            self.prompt_builder._events.append(last_event)

        if len(self.icl_episodes) > 0:
            if not self.cached_icl:
                messages = self.get_icl_prompt()
            else:
                messages = []
            messages.extend(self.prompt_builder.get_prompt(icl_episodes=True))
        else:
            messages = self.prompt_builder.get_prompt()

        # Add naive instructions to the last user message
        naive_instruction = """
You can only output one of the above actions at a time, and always have to output an action until the episode terminates.
Action:
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        return response
