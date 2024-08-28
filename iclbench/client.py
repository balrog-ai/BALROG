import os
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic


class LLMClientWrapper:
    def __init__(self, client_config):
        self.model_id = client_config.model_id
        self.api_key = client_config.api_key
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.is_chat_model = client_config.is_chat_model

        self.client_kwargs = {
            "model": client_config.model_id,
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
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("")

    def generate(self, input):
        response = self.model.generate_content(input, **self.client_kwargs)
        return response.generated_text


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


def create_llm_client(client_config):
    """
    Factory function to create the appropriate LLM client based on the model name.
    """
    if "gpt" in client_config.model_id:
        return OpenAIWrapper(client_config)
    elif "gemini" in client_config.model_id:
        return GoogleGenerativeAIWrapper(client_config)
    elif "claude" in client_config.model_id:
        return ClaudeWrapper(client_config)
    else:
        return OpenAIWrapper(client_config)
