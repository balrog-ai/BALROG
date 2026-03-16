import base64
import datetime
import logging
import time
import json
import csv
import os
from collections import namedtuple
from io import BytesIO

from google import genai
from google.genai import types

from anthropic import Anthropic
from openai import OpenAI

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
    ],
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """Base class for LLM client wrappers.

    Provides common functionality for interacting with different LLM APIs, including
    handling retries and common configuration settings. Subclasses should implement
    the `generate` method specific to their LLM API.
    """

    def __init__(self, client_config):
        """Initialize the LLM client wrapper with configuration settings.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.client_kwargs = {**client_config.generate_kwargs}
        self.max_retries = client_config.max_retries
        self.delay = client_config.delay
        self.alternate_roles = client_config.alternate_roles

    def generate(self, messages):
        """Generate a response from the LLM given a list of messages.

        This method should be overridden by subclasses.

        Args:
            messages (list): A list of messages to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retries upon failure.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function call.

        Raises:
            Exception: If the function fails after the maximum number of retries.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logger.error(f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}")
                sleep_time = self.delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)
        raise Exception(f"Failed to execute {func.__name__} after {self.max_retries} retries.")


def process_image_openai(image):
    """Process an image for OpenAI API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for OpenAI.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for OpenAI
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


def process_image_claude(image):
    """Process an image for Anthropic's Claude API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for Claude.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for Anthropic
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
    }


def process_image_bedrock(image):
    """Process an image for AWS Bedrock Converse API.

    Notes:
        The Bedrock Runtime Converse API accepts image content blocks with raw bytes.
        Not all models support images; users can disable image history if needed.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return {
        "image": {
            "format": "png",
            "source": {"bytes": buffered.getvalue()},
        }
    }


class OpenAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with the OpenAI API."""

    def __init__(self, client_config):
        """Initialize the OpenAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            elif self.client_name.lower() == "nvidia" or self.client_name.lower() == "xai":
                if not self.base_url or not self.base_url.strip():
                    raise ValueError("base_url must be provided when using NVIDIA or XAI client")
                self.client = OpenAI(base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                # For OpenAI, always use the standard API regardless of base_url
                self.client = OpenAI()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the OpenAI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the OpenAI API.
        """
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            if msg.attachment is not None:
                new_content.append(process_image_openai(msg.attachment))
            if self.alternate_roles and converted_messages and converted_messages[-1]["role"] == msg.role:
                converted_messages[-1]["content"].extend(new_content)
            else:
                converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages):
        """Generate a response from the OpenAI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the OpenAI API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )


class GoogleGenerativeAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with Google's Generative AI API."""

    def __init__(self, client_config):
        """Initialize the GoogleGenerativeAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Generative AI client if not already initialized."""
        if not self._initialized:
            self.client = genai.Client()
            self.model = None
            # Create kwargs dictionary for GenerationConfig
            client_kwargs = {
                "max_output_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                client_kwargs["temperature"] = temperature

            # Configure thinking / reasoning if an integer budget is provided.
            thinking_config = None
            thinking_budget = self.client_kwargs.get("thinking_budget", -1)
            if isinstance(thinking_budget, int):
                thinking_config = types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=True,
                )
            elif thinking_budget is not None:
                logger.warning(
                    "Ignoring non-integer thinking_budget=%r for Google Generative AI client; "
                    "expected an int number of tokens.",
                    thinking_budget,
                )

            if thinking_config is not None:
                self.generation_config = genai.types.GenerateContentConfig(
                    **client_kwargs,
                    thinking_config=thinking_config,
                )
            else:
                self.generation_config = genai.types.GenerateContentConfig(
                    **client_kwargs,
                )
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the new Google GenAI SDK.

        Args:
            messages (list): A list of message objects.

        Returns:
            list[types.Content]: A list of Content objects formatted for the API.
        """
        converted_messages = []
        
        for msg in messages:
            parts = []
            
            role = msg.role
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
                
            if msg.content:
                parts.append(types.Part(text=msg.content))

            if msg.attachment is not None:
                parts.append(types.Part(image=msg.attachment))

            converted_messages.append(
                types.Content(role=role, parts=parts)
            )
        return converted_messages

    def extract_completion(self, response):
        """Extract the completion text (answer) from the API response.

        This concatenates all non-thinking parts from the first candidate.

        Args:
            response: The response object from the API.

        Returns:
            str: The extracted completion text.

        Raises:
            Exception: If response is None or missing expected fields.
        """
        if not response:
            raise Exception("Response is None, cannot extract completion.")

        candidates = getattr(response, "candidates", [])
        if not candidates:
            raise Exception("No candidates found in the response.")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            raise Exception("No content found in the candidate.")

        content_parts = getattr(content, "parts", [])
        if not content_parts:
            raise Exception("No content parts found in the candidate.")

        answer_chunks = []
        for part in content_parts:
            text = getattr(part, "text", None)
            if not text:
                continue
            # Skip internal thinking / reasoning parts for the user-facing answer.
            if getattr(part, "thought", False):
                continue
            answer_chunks.append(text)

        if not answer_chunks:
            # Fallback to the response-level text helper if available.
            text = getattr(response, "text", None)
            if callable(text):
                text = text()
            if isinstance(text, str) and text.strip():
                return text.strip()
            raise Exception("No non-thinking text found in the content parts.")

        return "".join(answer_chunks).strip()

    def extract_reasoning(self, response):
        """Extract the reasoning / thinking trace from the API response, if present."""
        if not response:
            return None

        candidates = getattr(response, "candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            return None

        content_parts = getattr(content, "parts", [])
        if not content_parts:
            return None

        reasoning_chunks = []
        for part in content_parts:
            if not getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                reasoning_chunks.append(text.strip())

        if not reasoning_chunks:
            return None

        return "\n".join(reasoning_chunks)

    def _extract_reasoning_token_count(self, response) -> int:
        """Extract Gemini 'thinking/reasoning' token count from usage metadata.

        Gemini can report reasoning tokens separately from candidate/output tokens.
        Since downstream code tracks only input/output, we fold reasoning tokens into
        `output_tokens` so total accounting is correct.
        """
        usage = getattr(response, "usage_metadata", None) if response else None
        if usage is None:
            return 0

        # Different SDK versions / endpoints may use different names.
        for attr in (
            "thoughts_token_count",
            "thought_token_count",
            "thinking_token_count",
            "reasoning_token_count",
        ):
            val = getattr(usage, attr, None)
            if isinstance(val, int):
                return val
        return 0

    def generate(self, messages):
        """Generate a response from the Generative AI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the Generative AI API.
        """
        self._initialize_client()

        converted_messages = self.convert_messages(messages)

        def api_call():
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=converted_messages,
                config=self.generation_config,
            )
            # Attempt to extract completion and reasoning immediately after API call
            completion = self.extract_completion(response)
            reasoning = self.extract_reasoning(response)
            # Return both response and extracted fields if successful
            return response, completion, reasoning

        try:
            # Execute the API call and extraction together with retries
            response, completion, reasoning = self.execute_with_retries(api_call)
            reasoning_tokens = self._extract_reasoning_token_count(response)
            prompt_tokens = (
                getattr(response.usage_metadata, "prompt_token_count", 0)
                if response and getattr(response, "usage_metadata", None)
                else 0
            )
            candidate_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0)
                if response and getattr(response, "usage_metadata", None)
                else 0
            )
            output_tokens = (candidate_tokens or 0) + (reasoning_tokens or 0)

            # Check if the successful response contains an empty completion
            if not completion or completion.strip() == "":
                logger.warning(f"Gemini returned an empty completion for model {self.model_id}. Returning default empty response.")
                return LLMResponse(
                    model_id=self.model_id,
                    completion="",
                    stop_reason="empty_response",
                    input_tokens=prompt_tokens or 0,
                    output_tokens=output_tokens,
                    reasoning=reasoning,
                )
            else:
                # If completion is not empty, return the normal response
                return LLMResponse(
                    model_id=self.model_id,
                    completion=completion,
                    stop_reason=(
                        getattr(response.candidates[0], "finish_reason", "unknown")
                        if response and getattr(response, "candidates", [])
                        else "unknown"
                    ),
                    input_tokens=prompt_tokens or 0,
                    output_tokens=output_tokens,
                    reasoning=reasoning,
                )
        except Exception as e:
            logger.error(f"API call failed after {self.max_retries} retries: {e}. Returning empty completion.")
            # Return a default response indicating failure
            return LLMResponse(
                model_id=self.model_id,
                completion="",
                stop_reason="error_max_retries",
                input_tokens=0, # Assuming 0 tokens consumed if call failed
                output_tokens=0,
                reasoning=None,
            )


class ClaudeWrapper(LLMClientWrapper):
    """Wrapper for interacting with Anthropic's Claude API."""

    def __init__(self, client_config):
        """Initialize the ClaudeWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Claude client if not already initialized."""
        if not self._initialized:
            self.client = Anthropic()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Claude API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the Claude API.
        """
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
        """Generate a response from the Claude API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the Claude API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.messages.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.content[0].text.strip(),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning=None,
        )


class AWSBedrockWrapper(LLMClientWrapper):
    """Wrapper for interacting with Amazon AWS Bedrock Runtime Converse API."""

    def __init__(self, client_config):
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        if self._initialized:
            return

        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
        except Exception as e:
            raise ImportError(
                "AWS Bedrock client requires 'boto3' (and botocore). Install it and ensure AWS credentials are configured."
            ) from e

        # Allow passing AWS-specific settings via generate_kwargs for convenience.
        region_name = (
            self.client_kwargs.get("region_name")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        profile_name = self.client_kwargs.get("profile_name") or self.client_kwargs.get("aws_profile_name")

        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()

        endpoint_url = self.base_url.strip() if isinstance(self.base_url, str) and self.base_url.strip() else None

        botocore_config = BotocoreConfig(
            connect_timeout=self.timeout,
            read_timeout=self.timeout,
            retries={"max_attempts": max(1, int(self.max_retries)), "mode": "standard"},
        )

        self.client = session.client(
            "bedrock-runtime",
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=botocore_config,
        )
        self._initialized = True

    def convert_messages(self, messages):
        """Convert internal Message objects to Bedrock Converse message format."""
        converted_messages = []
        for msg in messages:
            role = msg.role
            if role == "system":
                # Keep compatibility with prompt builders that may emit system messages.
                role = "user"

            content_blocks = []
            if msg.content:
                content_blocks.append({"text": msg.content})
            if msg.attachment is not None:
                content_blocks.append(process_image_bedrock(msg.attachment))

            if (
                self.alternate_roles
                and converted_messages
                and converted_messages[-1]["role"] == role
            ):
                converted_messages[-1]["content"].extend(content_blocks)
            else:
                converted_messages.append({"role": role, "content": content_blocks})
        return converted_messages

    def _extract_text(self, response):
        try:
            blocks = response["output"]["message"]["content"]
        except Exception as e:
            raise KeyError(f"Unexpected Bedrock response shape: {response}") from e

        parts = []
        for block in blocks or []:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts).strip()

    def _extract_reasoning(self, response):
        """Extract extended thinking / reasoning content from Bedrock responses.

        Supports both Claude extended thinking (`type: thinking`) and Nova-style
        `reasoningContent` blocks when present.
        """
        try:
            blocks = response["output"]["message"]["content"]
        except Exception:
            return None

        reasoning_parts = []

        for block in blocks or []:
            if not isinstance(block, dict):
                continue

            # Claude 4.x extended thinking via Messages API (proxied by Converse).
            if block.get("type") == "thinking":
                thinking_text = block.get("thinking")
                if isinstance(thinking_text, str) and thinking_text.strip():
                    reasoning_parts.append(thinking_text.strip())

            # Amazon Nova extended reasoning via reasoningContent.
            reasoning_content = block.get("reasoningContent")
            if isinstance(reasoning_content, dict):
                reasoning_text = reasoning_content.get("reasoningText")
                if isinstance(reasoning_text, dict):
                    txt = reasoning_text.get("text")
                    if isinstance(txt, str) and txt.strip():
                        reasoning_parts.append(txt.strip())

        if not reasoning_parts:
            return None

        return "\n".join(reasoning_parts)

    def generate(self, messages):
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Map OpenAI-style kwargs to Bedrock Converse parameters.
            inference_config = {}
            additional_model_request_fields = {}

            max_tokens = self.client_kwargs.get("max_tokens", 1024)
            if max_tokens is not None:
                inference_config["maxTokens"] = int(max_tokens)

            thinking_budget = self.client_kwargs.get("thinking_budget")
            thinking_enabled = False
            if thinking_budget is not None:
                if isinstance(thinking_budget, int):
                    # Extended thinking for Claude 4.x models on Bedrock.
                    # Per AWS docs, this must be sent as a model-specific field.
                    additional_model_request_fields["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }
                    thinking_enabled = True
                else:
                    logger.warning(
                        "Ignoring non-integer thinking_budget=%r for AWS Bedrock client; "
                        "expected an int number of tokens.",
                        thinking_budget,
                    )

            # Extended thinking is not compatible with temperature/top_p tweaks.
            if not thinking_enabled:
                temperature = self.client_kwargs.get("temperature")
                if temperature is not None:
                    inference_config["temperature"] = float(temperature)

                top_p = self.client_kwargs.get("top_p")
                if top_p is not None:
                    inference_config["topP"] = float(top_p)

            return self.client.converse(
                modelId=self.model_id,
                messages=converted_messages,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_request_fields,
            )

        response = self.execute_with_retries(api_call)
        completion = self._extract_text(response)
        reasoning = self._extract_reasoning(response)

        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        input_tokens = usage.get("inputTokens", 0) or 0
        output_tokens = usage.get("outputTokens", 0) or 0

        stop_reason = response.get("stopReason") if isinstance(response, dict) else None

        return LLMResponse(
            model_id=self.model_id,
            completion=completion,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning=reasoning,
        )


def create_llm_client(client_config):
    """
    Factory function to create the appropriate LLM client based on the client name.

    Args:
        client_config: Configuration object containing client-specific settings.

    Returns:
        callable: A factory function that returns an instance of the appropriate LLM client.
    """

    def client_factory():
        client_name_lower = client_config.client_name.lower()
        if "openai" in client_name_lower or "vllm" in client_name_lower or "nvidia" in client_name_lower or "xai" in client_name_lower:
            # NVIDIA and XAI use OpenAI-compatible API, so we use the OpenAI wrapper
            return OpenAIWrapper(client_config)
        elif "gemini" in client_name_lower:
            return GoogleGenerativeAIWrapper(client_config)
        elif "claude" in client_name_lower:
            return ClaudeWrapper(client_config)
        elif "aws-bedrock" in client_name_lower:
            return AWSBedrockWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory
