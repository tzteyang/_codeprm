import httpx
import warnings
import asyncio
import tiktoken
import time
import random
import requests
from functools import wraps
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Dict, Optional, Type, List, Union, Awaitable, Tuple, Callable
)
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)
from enum import Enum, auto
from openai import OpenAI
from openai.types import Completion
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from openai import APITimeoutError, RateLimitError
from httpx import ConnectTimeout
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class GenerationConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __str__(self):
        return '\n'.join(f"{k} = {v}" for k, v in self.__dict__.items())

@dataclass
class vLLMConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __str__(self):
        return '\n'.join(f"{k} = {v}" for k, v in self.__dict__.items())



class ModelType(Enum):
    HF = "huggingface"
    OPENAI = "openai"
    VLLM = "vllm"
    VLLMSERVER = "vllm_server"


def retry_with_exponential_backoff(
    retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (
        ConnectTimeout,
        APITimeoutError,
        RateLimitError
    )
):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == retries - 1:
                        raise e
                    sleep_time = delay + random.uniform(0, delay)
                    print(f"Request failed, error: {e}. Retrying in {sleep_time:.2f} seconds (attempt {attempt + 1})")
                    time.sleep(sleep_time)
                    delay = min(delay * 2, max_delay)
                except Exception as e:
                    # raise e
                    return None
        return wrapper
    return decorator


class BaseLLM(ABC):
    def __init__(
            self, 
            model_name: str, 
            generation_config: GenerationConfig):
        self.model_name = model_name
        self.generation_config = generation_config
        self.model = None
        self.max_parallel_num = None
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        pass
    
    @abstractmethod
    def is_async(self) -> bool:
        pass
    
    @abstractmethod
    def num_tokens_from_string(self, string: str) -> int:
        pass

    # @abstractmethod
    def generate(
        self, 
        prompts: Optional[Union[str, List[str]]] = None, 
        messages: List[Dict] = None,
        prefix: Optional[Union[str, List[str]]] = None, 
    ) -> Union[str, Awaitable[str], List[str], Awaitable[List[str]]]:

        if self.is_async():
            return self._async_generate(prompts, messages, prefix)
        else:
            return self._sync_generate(prompts, messages, prefix)
    
    async def _async_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None, 
        messages: List[Dict] = None,
        prefix: Optional[Union[str, List[str]]] = None, 
    ) -> str:
        raise NotImplementedError("should be implemented in the child class")

    def _sync_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None, 
        messages: List[Dict] = None,
        prefix: Optional[Union[str, List[str]]] = None, 
    ) -> str:
        raise NotImplementedError("should be implemented in the child class")
            

class OpenAILLM(BaseLLM):
    def is_async(self) -> bool:
        return True

    def _initialize(self) -> None:
        self.client = OpenAI(
            base_url='https://api.xty.app/v1',
            api_key='sk-zGNulXpupOin3ses890f249dEf664348Ac10F7EbDbE87a05',
            http_client=httpx.Client(
                base_url='https://api.xty.app/v1',
                follow_redirects=True
            )
        )

        self.max_parallel_num = 16
        self.tokenizer = tiktoken.encoding_for_model(self.model_name) 
    
    def num_tokens_from_string(self, string: str) -> int:
        num_tokens = len(self.tokenizer.encode(string))
        return num_tokens
    
    @retry_with_exponential_backoff()
    def _create_completion(self, messages, max_tokens, max_completion_tokens):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=getattr(self.generation_config, 'temperature', 0.7),
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens
        )

    async def _async_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None, 
        messages: Optional[List[Dict]] = None,
        prefix: Optional[str] = None
    ) -> str:
        if prefix is not None:
            warnings.warn(
                "OpenAI api model not support `prefix` args, will be added at the end of user's content"
            )
        
        assert prompts is None or messages is None, "Do not supply parameters `prompts` and `messages` together"

        if messages is None:
            prompts += prefix if prefix is not None else ''
            messages = [
                {"role": "user", "content": prompts}
            ]

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            max_completion_tokens = getattr(self.generation_config, 'max_completion_tokens', None)
            max_tokens = getattr(self.generation_config, 'max_tokens', 512) if max_completion_tokens is None else None

            completion = await loop.run_in_executor(
                pool,
                self._create_completion,
                messages,
                max_tokens,
                max_completion_tokens
            )
            # completion = await loop.run_in_executor(pool, create_completion)

        response = completion.choices[0].message.content if completion is not None else completion
        return response


class vLLM(BaseLLM):
    def is_async(self) -> bool:
        return False

    def _initialize(self) -> None:
        self.max_parallel_num = 16
        self.config = getattr(self.generation_config, 'vllm_config', None)
        if self.config is None:
            raise ValueError('vLLM config is required for vLLM model')
        self.tensor_parallel_size = getattr(self.config, 'tensor_parallel_size', 1)

        print(f"Loading vLLM model '{self.model_name}'...")
        self.llm = LLM(self.model_name, tensor_parallel_size=self.tensor_parallel_size)
        print("vLLM model loaded successfully.")

        self.tokenizer = self.llm.get_tokenizer()

    def num_tokens_from_string(self, string: str) -> int:
        num_tokens = len(self.tokenizer.encode(string))
        return num_tokens

    def _sync_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None, 
        messages: Optional[List[Dict]] = None,
        prefix: Optional[str] = None
    ) -> str:
        assert prompts is None or messages is None, "Do not supply parameters `prompts` and `messages` together"
        sampling_params_kwargs = {
            'temperature': getattr(self.generation_config, 'temperature', None),
            'max_tokens': getattr(self.generation_config, 'max_tokens', None),
            'top_p': getattr(self.generation_config, 'top_p', None),
            'repetition_penalty': getattr(self.generation_config, 'repetition_penalty', None)
        }
        sampling_params = SamplingParams(
            **{k: v for k, v in sampling_params_kwargs.items() if v is not None}
        )
        if isinstance(prompts, str):
            prompts = [prompts]
        is_chat = getattr(self.generation_config, 'is_chat', False)
        if is_chat:
            if messages is None:
                _add_generation_prompt = True if prefix is None else False
                messages = [
                    [{"role": "user", "content": prompt}] \
                        if prefix is None else \
                    [{"role": "user", "content": prompt}, {"role": "assistant", "content": prefix}]
                    for prompt in prompts
                ]
            input_texts = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=_add_generation_prompt
            )
            input_texts = [text.removesuffix(self.tokenizer.eos_token + '\n') for text in input_texts]
        else:
            if messages is not None:
                raise ValueError('`messages` params no need when not in chat mode')
            input_texts = [prompt if prefix is None else prompt + prefix for prompt in prompts]
            
        outputs = self.llm.generate(prompts=input_texts, sampling_params=sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts


class vLLMServer(BaseLLM):
    def is_async(self) -> bool:
        return True
    
    def _initialize(self) -> None:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type((requests.RequestException, Exception)),
            before_sleep=lambda retry_state: print(
                f"Attempt {retry_state.attempt_number} failed, {retry_state.outcome.exception()}, "
                f"retrying in {retry_state.next_action.sleep} seconds..."
            ),
            retry_error_callback=lambda retry_state: (False, str(retry_state.outcome.exception()))
        )
        def _check_api_health(url: str, timeout: int = 5) -> Tuple[bool, Optional[str]]:
            """
            Checks the health of an API by sending a GET request to the specified URL.

            Args:
                url (str): The URL of the API to check.
                timeout (int, optional): The timeout for the GET request in seconds. Defaults to 5.

            Returns:
                Tuple[bool, Optional[str]]: A tuple where the first element is a boolean indicating
                whether the API is healthy (True) or not (False), and the second element is an optional
                string containing the error message if the API is not healthy.
            """
            headers = {
                "Accept": "application/json",
            }

            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return True, None
            
        self.config = getattr(self.generation_config, 'vllm_config', None)
        if self.config is None:
            raise ValueError('vLLM config is required for vLLM model')
        
        host, port, self.api_key = getattr(self.config, 'host', 'localhost'), \
            getattr(self.config, 'port', 8000), \
            getattr(self.config, 'api_key', 'EMPTY_API_KEY')
        
        self.base_url = f'http://{host}:{port}'
        print('Checking vLLM API health...')
        is_healthy, error_msg = _check_api_health(f'{self.base_url}/health')
        if not is_healthy:
            raise Exception(f'vLLM API is not healthy: {error_msg}')
        print('vLLM API is healthy and ready for any requests!')

        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url+'/v1',
        )
        print(f'vLLM API wrapper created with model: {self.client.models.list().data[0].id}')

        self.max_parallel_num = 8
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def num_tokens_from_string(self, string: str) -> int:
        num_tokens = len(self.tokenizer.encode(string))
        return num_tokens

    @retry_with_exponential_backoff()
    def _create_completion(
        self, 
        model_name: str,
        prompts: Union[str, List[str]], 
        **kwargs
    ) -> Completion:
        return self.client.completions.create(
            model=model_name,
            prompt=prompts,
            **{k: v for k, v in kwargs.items() if v is not None}
        )

    async def _async_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None, 
        messages: Optional[List[Dict]] = None,
        prefix: Optional[str] = None
    ) -> str:
        assert prompts is None or messages is None, "Do not supply parameters `prompts` and `messages` together"

        is_chat = getattr(self.generation_config, 'is_chat', False)
        if is_chat:
            if messages is None:
                _add_generation_prompt = True if prefix is None else False
                messages = [
                    [{"role": "user", "content": prompt}] \
                        if prefix is None else \
                    [{"role": "user", "content": prompt}, {"role": "assistant", "content": prefix}]
                    for prompt in prompts
                ]
            input_texts = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=_add_generation_prompt
            )
            input_texts = [text.removesuffix(self.tokenizer.eos_token + '\n') for text in input_texts]
        else:
            if messages is not None:
                raise ValueError('`messages` params no need when not in chat mode')
            input_texts = [prompt if prefix is None else prompt + prefix for prompt in prompts]
        # breakpoint()
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            completion = await loop.run_in_executor(
                pool,
                partial(
                    self._create_completion,
                    model_name=self.model_name,
                    prompts=input_texts,
                    temperature=getattr(self.generation_config, 'temperature', None),
                    max_tokens=getattr(self.generation_config, 'max_tokens', None),
                    top_p=getattr(self.generation_config, 'top_p', None),
                    top_k=getattr(self.generation_config, 'top_k', None),
                    repetition_penalty=getattr(self.generation_config, 'repetition_penalty', None)
                )
            )

        responses = [choice.text for choice in completion.choices]

        return responses


class LLMFactory:
    _models: Dict[ModelType, Type[BaseLLM]] = {
        # ModelType.HF: HuggingFaceLLM,
        ModelType.OPENAI: OpenAILLM,
        ModelType.VLLM: vLLM,
        ModelType.VLLMSERVER: vLLMServer
    }

    @classmethod
    def create(
        cls,
        model_type: str,
        model_name: str,
        generation_config: GenerationConfig
    ) -> BaseLLM:
        try:
            model_type = ModelType(model_type.lower())

            model_class = cls._models.get(model_type)
            if not model_class:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            return model_class(model_name, generation_config)
        
        except Exception as e:
            import traceback
            raise RuntimeError(f"Failed to create LLM instance: {traceback.format_exc()}") 