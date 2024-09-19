from collections import deque
from typing import List, Optional


class Message:
    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class HistoryPromptBuilder:
    def __init__(self, max_history: int = 16, max_image_history: int = 1, system_prompt: Optional[str] = None):
        self.max_history = max_history
        self.max_image_history = min(max_image_history, max_history)
        self.system_prompt = system_prompt
        self._events = deque(maxlen=max_history * 2)  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation

    def update_instruction_prompt(self, instruction: str):
        self.system_prompt = instruction

    def update_observation(self, obs: dict):
        # Extract text and image from the observation
        long_term_context = obs["text"].get("long_term_context", "")
        self._last_short_term_obs = obs["text"].get("short_term_context", "")
        text = long_term_context
        if self._last_short_term_obs:
            text = f"Current Observation:\n{self._last_short_term_obs}\n{text}"
        image = obs.get("image", None)

        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "image": image,
            }
        )

    def update_action(self, action: str):
        # Add action to events
        self._events.append(
            {
                "type": "action",
                "action": action,
            }
        )

    def reset(self):
        self._events.clear()

    def get_prompt(self) -> List[Message]:
        messages = []

        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))

        # Determine which images to include
        images_needed = self.max_image_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if images_needed > 0 and event.get("image") is not None:
                    event["include_image"] = True
                    images_needed -= 1
                else:
                    event["include_image"] = False

        # Process events to create messages
        for event in self._events:
            if event["type"] == "observation":
                content = event["text"]
                image = event.get("image") if event.get("include_image", False) else None
                message = Message(role="user", content=content, attachment=image)
                # Clean up the temporary flag
                if "include_image" in event:
                    del event["include_image"]
            elif event["type"] == "action":
                content = event["action"]
                message = Message(role="assistant", content=content)
            messages.append(message)

        return messages
