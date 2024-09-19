import base64
from collections import deque
from io import BytesIO


def process_image_openai(image):
    # Encode the image as a base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for OpenAI
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


def process_image_gemini(image):
    # For Gemini, include the image directly
    return image


# Model configuration dictionary
MODEL_CONFIG = {
    "openai": {
        "system_name": "system",
        "assistant_name": "assistant",
        "content_name": "content",
        "process_image": process_image_openai,
    },
    "gemini": {
        "system_name": "user",
        "assistant_name": "model",
        "content_name": "parts",
        "process_image": process_image_gemini,
    },
    "claude": {
        "system_name": "user",
        "assistant_name": "assistant",
        "content_name": "content",
        "process_image": process_image_openai,
    },
}


class HistoryPromptBuilder:
    def __init__(
        self,
        *,
        max_history=16,
        max_image_history=1,
        system_prompt=None,
        model_type="claude",
        chat_history=True,
        sep="_" * 80,
    ):
        self._max_history = max_history
        self.max_image_history = min(max_image_history, max_history)
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.chat_history = chat_history
        self.sep = sep

        model_config = MODEL_CONFIG.get(model_type, {})
        self.system_role = model_config["system_name"]
        self.assistant_role = model_config["assistant_name"]
        self.content_name = model_config["content_name"]
        self.process_image = model_config["process_image"]

        self._events = deque(maxlen=self._max_history * 2)
        self._last_short_term_obs = None

    def build_content(self, text_content, processed_image):
        if processed_image is not None and not isinstance(processed_image, list):
            processed_image = [processed_image]

        if self.model_type == "openai":
            # OpenAI expects a list of content elements
            content = []
            if text_content:
                content.append({"type": "text", "text": text_content})
            if processed_image:
                content.extend(processed_image)
            return content
        elif self.model_type == "gemini":
            # Gemini uses 'parts' with text and images
            content = []
            if text_content:
                content.append(text_content)
            if processed_image:
                content.extend(processed_image)
            return content
        else:
            # Default behavior: content is just text
            return text_content

    def format_message(self, role, content):
        return {
            "role": role,
            self.content_name: content,
        }

    def update_observation(self, obs):
        self._last_short_term_obs = obs["text"].get("short_term_context", "")
        long_term_context = obs["text"].get("long_term_context", "")

        image = obs.get("image", None)
        if image is not None and self.process_image and self.max_image_history > 0:
            processed_image = self.process_image(image)
        else:
            processed_image = None

        event = {
            "type": "observation",
            "text": long_term_context,
            "image": processed_image,
        }
        self._events.append(event)

    def update_action(self, action):
        event = {
            "type": "action",
            "action": action,
        }
        self._events.append(event)

    def update_instruction_prompt(self, prompt):
        self.system_prompt = prompt

    def reset(self):
        self._events.clear()

    def get_prompt(self):
        if self.chat_history:
            return self.get_chat_history()
        else:
            return self.get_completion_history()

    def get_completion_history(self):
        # Not working yet
        images = []

        history = "\n\nObservation history\n"

        # Annotate the last N observations with images
        images_needed = self.max_image_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if images_needed > 0 and event.get("image") is not None:
                    event["include_image"] = True
                    images_needed -= 1
                else:
                    event["include_image"] = False

        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                image = event["image"] if event.get("include_image") else None
                if image is not None:
                    images.append(event["image"])
                if idx == len(self._events) - 1:
                    history += (
                        self.sep + "\n" + "Current Observation:\n" + self._last_short_term_obs + "\n" + event["text"]
                    )
                else:
                    history += self.sep + "\n" + event["text"]
            elif event["type"] == "action":
                history += "\n\nAction: " + event["action"] + "\n"

        prompt = self.system_prompt + history + "\n\n" + "Next action:"
        content = self.build_content(prompt, images)
        return self.format_message("user", content)

    def get_chat_history(self):
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": self.system_role,
                    self.content_name: self.system_prompt,
                }
            )

        # Annotate the last N observations with images
        images_needed = self.max_image_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if images_needed > 0 and event.get("image") is not None:
                    event["include_image"] = True
                    images_needed -= 1
                else:
                    event["include_image"] = False

        # Build the chat history
        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                image = event["image"] if event.get("include_image") else None
                image_obs = "\nObservation Image: " if image is not None else ""
                if idx == len(self._events) - 1:
                    # Add short term context to the last observation (NLE/Craftax/MiniHack)
                    content = self.build_content(
                        "Current Observation:\n" + self._last_short_term_obs + "\n" + event["text"] + image_obs,
                        image,
                    )
                else:
                    content = self.build_content("Obesrvation:\n" + event["text"] + image_obs, image)
                message = self.format_message("user", content)
                del event["include_image"]
            elif event["type"] == "action":
                content = event["action"]
                if self.model_type == "openai":
                    content = [{"type": "text", "text": content}]
                elif self.model_type == "gemini":
                    content = [content]
                message = self.format_message(self.assistant_role, content)
            messages.append(message)

        return messages
