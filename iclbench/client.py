import base64
from collections import namedtuple
from io import BytesIO

import google.generativeai as genai
import replicate
from anthropic import Anthropic
from openai import OpenAI

LLMResponse = namedtuple(
    "LLMResponse", ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"]
)


class LLMClientWrapper:
    def __init__(self, client_config):
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.is_chat_model = client_config.is_chat_model
        self.client_kwargs = {**client_config.generate_kwargs}

    def generate(self, messages):
        raise NotImplementedError("This method should be overridden by subclasses")


def process_image_openai(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for OpenAI
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


def process_image_claude(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for Anthropic
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}}


class OpenAIWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                self.client = OpenAI()
            self._initialized = True

    def convert_messages(self, messages):
        converted_messages = []
        for msg in messages:
            converted_messages.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
            if msg.attachment is not None:
                converted_messages[-1]["content"].append(process_image_openai(msg.attachment))

        return converted_messages

    def generate(self, messages):
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        response = self.client.chat.completions.create(
            messages=converted_messages,
            model=self.model_id,
            max_tokens=self.client_kwargs.get("max_tokens", 1024),
        )

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )


class GoogleGenerativeAIWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        if not self._initialized:
            self.model = genai.GenerativeModel(self.model_id)

            client_kwargs = {
                "temperature": self.client_kwargs.get("temperature", 0.5),
                "max_output_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            self.generation_config = genai.types.GenerationConfig(**client_kwargs)
            self._initialized = True

    def convert_messages(self, messages):
        # Convert standard Message objects to Gemini's format
        converted_messages = []
        for msg in messages:
            parts = []
            if msg.role == "assistant":
                msg.role = "model"
            if msg.role == "system":
                msg.role = "user"
            if msg.content:
                parts.append(msg.content)
            if msg.attachment is not None:
                parts.append(msg.attachment)
            converted_messages.append(
                {
                    "role": msg.role,
                    "parts": parts,
                }
            )
        return converted_messages

    def generate(self, messages):
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        response = self.model.generate_content(
            converted_messages,
            generation_config=self.generation_config,
        )

        completion = (
            response.candidates[0].content.parts[0].text.strip()
            if len(response.candidates[0].content.parts) > 0
            else ""
        ).strip()

        return LLMResponse(
            model_id=self.model_id,
            completion=completion,
            stop_reason=response.candidates[0].finish_reason,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
            reasoning=None,
        )


class ClaudeWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        if not self._initialized:
            self.client = Anthropic()
            self._initialized = True

    def convert_messages(self, messages):
        converted_messages = []
        for msg in messages:
            converted_messages.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
            if converted_messages[-1]["role"] == "system":
                # Claude doesn't support system prompt and requires alternating roles
                converted_messages[-1]["role"] = "user"
                converted_messages.append({"role": "assistant", "content": "I'm ready!"})
            if msg.attachment is not None:
                converted_messages[-1]["content"].append(process_image_claude(msg.attachment))

        return converted_messages

    def generate(self, messages):
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        response = self.client.messages.create(
            messages=converted_messages,
            model=self.model_id,
            max_tokens=self.client_kwargs.get("max_tokens", 1024),
        )

        return LLMResponse(
            model_id=self.model_id,
            completion=response.content[0].text.strip(),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning=None,
        )


class ReplicateWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self.client = replicate.Client(timeout=self.timeout)

    def generate(self, messages):
        # Replicate models might not support multi-turn conversations; we concatenate messages
        prompt = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in messages])
        output = self.client.run(self.model_id, input={"prompt": prompt}, **self.client_kwargs)

        # Handle different output types
        if isinstance(output, list):
            content = "".join(output)
        elif isinstance(output, str):
            content = output
        else:
            content = str(output)

        return LLMResponse(
            model_id=self.model_id,
            completion=content.strip(),
            stop_reason=None,
            input_tokens=None,
            output_tokens=None,
            reasoning=None,
        )


def create_llm_client(client_config):
    """
    Factory function to create the appropriate LLM client based on the client name.
    """

    def client_factory():
        if "openai" in client_config.client_name.lower() or "vllm" in client_config.client_name.lower():
            return OpenAIWrapper(client_config)
        elif "gemini" in client_config.client_name.lower():
            return GoogleGenerativeAIWrapper(client_config)
        elif "claude" in client_config.client_name.lower():
            return ClaudeWrapper(client_config)
        elif "replicate" in client_config.client_name.lower():
            return ReplicateWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory
