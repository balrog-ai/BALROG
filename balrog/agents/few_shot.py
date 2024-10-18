from typing import List, Optional

from balrog.agents.base import BaseAgent


class Message:
    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class FewShotAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.icl_episodes = []
        self.icl_events = []
        self.cached_icl = False

    def update_icl_observation(self, obs: dict):
        long_term_context = obs["text"].get("long_term_context", "")
        self.icl_events.append(
            {
                "type": "icl_observation",
                "text": long_term_context,
            }
        )

    def update_icl_action(self, action: str):
        self.icl_events.append(
            {
                "type": "icl_action",
                "action": action,
            }
        )

    def cache_icl(self):
        self.client.cache_icl_demo(self.get_icl_prompt())
        self.cached_icl = True

    def wrap_episode(self):
        icl_episode = []
        icl_episode.append(
            Message(role="user", content=f"****** START OF DEMONSTRATION EPISODE {len(self.icl_episodes) + 1} ******")
        )
        for event in self.icl_events:
            if event["type"] == "icl_observation":
                content = "Obesrvation:\n" + event["text"]
                message = Message(role="user", content=content)
            elif event["type"] == "icl_action":
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

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        if not self.cached_icl:
            messages = self.get_icl_prompt()
        else:
            messages = []

        messages.extend(self.prompt_builder.get_prompt(icl_episodes=True))

        # Add naive instructions to the last user message
        naive_instruction = """
You can only output one of the above actions at a time, and always have to output an action until the episode terminates.
Action:
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        return response
