import copy
import re
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
    def __init__(self, client_factory, prompt_builder, max_icl_history):
        """Initialize the FewShotAgent with a client and prompt builder."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.icl_episodes = []
        self.icl_events = []
        self.max_icl_history = max_icl_history
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
        self.icl_events = []

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
        i = 0
        for icl_episode in self.icl_episodes:
            episode_steps = len(icl_episode) - 2  # not count start and end messages
            if i + episode_steps <= self.max_icl_history:
                icl_messages.extend(icl_episode)
                i += episode_steps
            else:
                icl_episode = icl_episode[: self.max_icl_history - i + 1] + [
                    icl_episode[-1]
                ]  # +1 for start message -1 for end message
                icl_messages.extend(icl_episode)
                i += len(icl_episode) - 2  # not count start and end messages
                break

        end_demo_message = Message(
            role="user",
            content="****** Now it's your turn to play the game! ******",
        )
        icl_messages.append(end_demo_message)

        return icl_messages

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        if not self.cached_icl:
            messages = self.get_icl_prompt()
        else:
            messages = []

        messages.extend(self.prompt_builder.get_prompt(icl_episodes=True))

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        final_answer = self._extract_final_answer(response)

        return final_answer

    def _extract_final_answer(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=filter_letters(final_answer.completion))

        return final_answer
