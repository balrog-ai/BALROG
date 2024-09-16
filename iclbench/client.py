from openai import OpenAI
import replicate
import google.generativeai as genai
from anthropic import Anthropic
from types import SimpleNamespace


class LLMClientWrapper:
    def __init__(self, client_config):
        self.model_id = client_config.model_id
        self.api_key = client_config.api_key
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.is_chat_model = client_config.is_chat_model

        self.client_kwargs = {
            **client_config.generate_kwargs,
        }

    def generate(self, input):
        raise NotImplementedError("This method should be overridden by subclasses")


class OpenAIWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        self.client_kwargs["model"] = self.model_id

    def generate(self, input):
        if self.is_chat_model and isinstance(input, list):  # Chat-based input
            completion = self.client.chat.completions.create(
                **self.client_kwargs, messages=input
            )
        else:  # Text-based input
            completion = self.client.completions.create(
                prompt=input[0]["content"], **self.client_kwargs
            )
        return completion


class GoogleGenerativeAIWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self._initialized = False  # Flag to check if client is initialized

    def _initialize_client(self):
        if not self._initialized:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_id)
            self.generation_config = genai.types.GenerationConfig(**self.client_kwargs)
            self._initialized = True

    def generate(self, input):
        self._initialize_client()  # Ensure client is initialized in child process
        response = self.model.generate_content(
            input[0]["content"], generation_config=self.generation_config
        )

        choices = [
            SimpleNamespace(
                index=idx,
                message=SimpleNamespace(
                    content=(
                        candidate.content.parts[0].text.strip()
                        if candidate.content.parts
                        else ""
                    ),
                    role="assistant",
                ),
            )
            for idx, candidate in enumerate(response.candidates)
        ]
        return SimpleNamespace(choices=choices)


class ClaudeWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self.client = Anthropic(api_key=self.api_key)

    def generate(self, input):
        if self.is_chat_model and isinstance(input, list):  # Chat-based input
            completion = self.client.chat.completions.create(
                **self.client_kwargs, messages=input
            )
        else:  # Text-based input
            completion = self.client.completions.create(
                prompt=input[0]["content"], **self.client_kwargs
            )
        return completion


class ReplicateWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self.client = replicate.Client(api_token=self.api_key, timeout=self.timeout)

    def generate(self, input):
        output = self.client.run(
            self.model_id, input={"prompt": input[0]["content"]}, **self.client_kwargs
        )

        # Handle different output types
        if isinstance(output, list):
            content = "".join(output)
        elif isinstance(output, str):
            content = output
        else:
            content = str(output)

        choices = [
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    content=content.strip(),
                    role="assistant",
                ),
            )
        ]
        completion = SimpleNamespace(choices=choices)

        return completion


def create_llm_client(client_config):
    """
    Factory function to create the appropriate LLM client based on the model name.
    """

    def client_factory():
        if "gpt" in client_config.client_name:
            return OpenAIWrapper(client_config)
        elif "gemini" in client_config.client_name:
            return GoogleGenerativeAIWrapper(client_config)
        elif "claude" in client_config.client_name:
            return ClaudeWrapper(client_config)
        elif "replicate" in client_config.client_name:
            return ReplicateWrapper(client_config)
        else:
            return OpenAIWrapper(client_config)

    return client_factory
