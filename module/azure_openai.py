import functools
import json
import logging
from typing import Any, Callable, Literal, Optional, cast

import backoff
import openai
import httpx

#from dspy import dsp
from module.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from abc import ABC, abstractmethod
import module.settings as settings


try:
    OPENAI_LEGACY = int(openai.version.__version__[0]) == 0
except Exception:
    OPENAI_LEGACY = True

try:
    import openai.error
    from openai.openai_object import OpenAIObject

    ERRORS = (
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
    )
except Exception:
    ERRORS = (openai.RateLimitError, openai.APIError)
    OpenAIObject = dict


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


AzureADTokenProvider = Callable[[], str]

class LM(ABC):
    """Abstract class for language models."""

    def __init__(self, model):
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        self.provider = "default"

        self.history = []

    @abstractmethod
    def basic_request(self, prompt, **kwargs):
        pass

    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def print_green(self, text: str, end: str = "\n"):
        import dspy

        if dspy.settings.experimental:
            return "\n\n" + "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end
        else:
            return "\x1b[32m" + str(text) + "\x1b[0m" + end

    def print_red(self, text: str, end: str = "\n"):
        return "\x1b[31m" + str(text) + "\x1b[0m" + end

    def inspect_history(self, n: int = 1, skip: int = 0):
        """Prints the last n prompts and their completions.

        TODO: print the valid choice that contains filled output field instead of the first.
        """
        provider: str = self.provider

        last_prompt = None
        printed = []
        n = n + skip

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:
                if provider in (
                    "clarifai",
                    "cloudflare",
                    "google",
                    "groq",
                    "Bedrock",
                    "Sagemaker",
                    "premai",
                    "tensorrt_llm",
                ):
                    printed.append((prompt, x["response"]))
                elif provider == "anthropic":
                    blocks = [
                        {"text": block.text}
                        for block in x["response"].content
                        if block.type == "text"
                    ]
                    printed.append((prompt, blocks))
                elif provider == "cohere":
                    printed.append((prompt, x["response"].text))
                elif provider == "mistral":
                    printed.append((prompt, x["response"].choices))
                elif provider == "ibm":
                    printed.append((prompt, x))
                elif provider == "you.com":
                    printed.append((prompt, x["response"]["answer"]))
                else:
                    printed.append((prompt, x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        printing_value = ""
        for idx, (prompt, choices) in enumerate(reversed(printed)):
            # skip the first `skip` prompts
            if (n - idx - 1) < skip:
                continue
            printing_value += "\n\n\n"
            printing_value += prompt

            text = ""
            if provider in (
                "cohere",
                "Bedrock",
                "Sagemaker",
                "clarifai",
                "claude",
                "ibm",
                "premai",
                "you.com",
                "tensorrt_llm",
            ):
                text = choices
            elif provider == "openai" or provider == "ollama":
                text = " " + self._get_choice_text(choices[0]).strip()
            elif provider == "groq":
                text = " " + choices
            elif provider == "google":
                text = choices[0].parts[0].text
            elif provider == "mistral":
                text = choices[0].message.content
            elif provider == "cloudflare":
                text = choices[0]
            else:
                text = choices[0]["text"]
            printing_value += self.print_green(text, end="")

            if len(choices) > 1 and isinstance(choices, list):
                printing_value += self.print_red(
                    f" \t (and {len(choices)-1} other completions)", end="",
                )

            printing_value += "\n\n\n"

        print(printing_value)
        return printing_value

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(model=model, **kwargs)

class AzureOpenAI(LM):
    """Wrapper around Azure's API for OpenAI.

    Args:
        api_base (str): Azure URL endpoint for model calling, often called 'azure_endpoint'.
        api_version (str): Version identifier for API.
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "gpt-3.5-turbo-instruct".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        http_client
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "chat".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        #api_base: str,
        api_version: str,
        tud_dev: str,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        azure_ad_token_provider: Optional[AzureADTokenProvider] = None,
        **kwargs,
    ):
        super().__init__(model_name)
        self.provider = "openai"

        self.system_prompt = system_prompt

        if tud_dev == 'TUD':
            llm_model_dict = {'gpt-35-turbo': '015442bdaallmsvc0101gpt35swctudopenai',
                            'gpt-35-turbo-16k': '015442bdaallmsvc0101gpt3516kswctudopenai',
                            'gpt-4': '015442bdaallmsvc0101gpt4swctudopenai',
                            'gpt-4-32k': '015442bdaallmsvc0101gpt432kswctudopenai',
                            'gpt-4o': '015442bdaallmsvc0101gpt4o20240513swctudopenaiannotated' 
                            }
        elif tud_dev == 'DEV':
            llm_model_dict = {'gpt-35-turbo-16k': '773814b04e6608aa1edf0db50321cc3a',
                            'gpt-35-turbo Sweden Central': 'ec38e691f34c53d7a5f1d1febf627514',
                            'gpt-4': 'a0e1112bfeaab98b7054b3da49fb05c',
                            'gpt-4-32k Sweden Central': '6aea959a626aba0033a5d3c0a638873c',
                            'gpt-4o': '19498621c089d7959382ddca714112a1' #wrong
                            }

        # LLM

        if tud_dev == 'TUD':
            api_base = 'https://015442-bdaallmsvc01-01-swc-tud-openai.privatelink.openai.azure.com'
        elif tud_dev == 'DEV':
            api_base = 'https://10.124.192.108/azr/' 


        model = llm_model_dict[model_name]

        # Define Client
        if OPENAI_LEGACY:
            # Assert that all variables are available
            assert (
                "engine" in kwargs or "deployment_id" in kwargs
            ), "Must specify engine or deployment_id for Azure API instead of model."

            openai.api_base = api_base
            openai.api_key = api_key
            openai.http_client = httpx.Client(verify=False)
            openai.api_type = "azure"
            openai.api_version = api_version
            openai.azure_ad_token_provider = azure_ad_token_provider

            self.client = None

        else:
            client = openai.AzureOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                http_client = httpx.Client(verify=False),
                api_version=api_version,
                azure_ad_token_provider=azure_ad_token_provider,
            )

            self.client = client

        self.model_type = model_type

        if not OPENAI_LEGACY and "model" not in kwargs:
            if "deployment_id" in kwargs:
                kwargs["model"] = kwargs["deployment_id"]
                del kwargs["deployment_id"]

            if "api_version" in kwargs:
                del kwargs["api_version"]

        if "model" not in kwargs:
            kwargs["model"] = model

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 1500,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.http_client = httpx.Client(verify=False)
        self.tud_dev = tud_dev
        self.model_name = model_name

        self.history: list[dict[str, Any]] = []

    def _openai_client(self):
        if OPENAI_LEGACY:
            return openai

        return self.client

    def log_usage(self, response):
        """Log the total tokens from the Azure OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logging.debug(f"Azure OpenAI Total Token Usage: {total_tokens}")

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            # caching mechanism requires hashable kwargs
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            kwargs["messages"] = messages
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = chat_request(self.client, **kwargs)

        else:
            kwargs["prompt"] = prompt
            response = completions_request(self.client, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of GPT-3 completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.model_type == "chat":
            return choice["message"]["content"]
        return choice["text"]

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from OpenAI Model.

        Args:
            prompt (str): prompt to send to GPT-3
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.request(prompt, **kwargs)

        self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(
            tud_dev = self.tud_dev,
            api_version = self.api_version, 
            model_name = self.model_name, 
            api_key = self.api_key,
            model_type = self.model_type,
            **kwargs,
        )


@CacheMemory.cache
def cached_gpt3_request_v2(**kwargs):
    return openai.Completion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gpt3_request_v2_wrapped(**kwargs):
    return cached_gpt3_request_v2(**kwargs)


@CacheMemory.cache
def _cached_gpt3_turbo_request_v2(**kwargs) -> OpenAIObject:
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    return cast(OpenAIObject, openai.ChatCompletion.create(**kwargs))


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def _cached_gpt3_turbo_request_v2_wrapped(**kwargs) -> OpenAIObject:
    return _cached_gpt3_turbo_request_v2(**kwargs)


def v1_chat_request(client, **kwargs):
    @functools.lru_cache(maxsize=None if cache_turn_on else 0)
    @NotebookCacheMemory.cache
    def v1_cached_gpt3_turbo_request_v2_wrapped(**kwargs):
        @CacheMemory.cache
        def v1_cached_gpt3_turbo_request_v2(**kwargs):
            if "stringify_request" in kwargs:
                kwargs = json.loads(kwargs["stringify_request"])
            return client.chat.completions.create(**kwargs)

        return v1_cached_gpt3_turbo_request_v2(**kwargs)

    return v1_cached_gpt3_turbo_request_v2_wrapped(**kwargs).model_dump()


def v1_completions_request(client, **kwargs):
    @functools.lru_cache(maxsize=None if cache_turn_on else 0)
    @NotebookCacheMemory.cache
    def v1_cached_gpt3_request_v2_wrapped(**kwargs):
        @CacheMemory.cache
        def v1_cached_gpt3_request_v2(**kwargs):
            return client.completions.create(**kwargs)

        return v1_cached_gpt3_request_v2(**kwargs)

    return v1_cached_gpt3_request_v2_wrapped(**kwargs).model_dump()


def chat_request(client, **kwargs):
    if OPENAI_LEGACY:
        return _cached_gpt3_turbo_request_v2_wrapped(**kwargs)

    return v1_chat_request(client, **kwargs)


def completions_request(client, **kwargs):
    if OPENAI_LEGACY:
        return cached_gpt3_request_v2_wrapped(**kwargs)

    return v1_completions_request(client, **kwargs)
