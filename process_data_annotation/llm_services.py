import httpx
import warnings
import asyncio
import tiktoken
import time
import random
from functools import wraps
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Dict, Optional, Type, List, Union, Awaitable, Tuple, Callable
)
from enum import Enum, auto
from openai import OpenAI
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from openai import APITimeoutError, RateLimitError
from httpx import ConnectTimeout
from vllm import LLM, SamplingParams


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
            generation_config: GenerationConfig, 
            vllm_config: Optional[vLLMConfig] = None):
        self.model_name = model_name
        self.generation_config = generation_config
        self.vllm_config = vllm_config
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
    ) -> Union[str, Awaitable[str]]:

        if self.is_async:
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

    def _sync_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None, 
        messages: Optional[List[Dict]] = None,
        prefix: Optional[str] = None
    ) -> str:

        return asyncio.run(self._async_generate(prompts, messages, prefix))
    
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



def vLLM(BaseLLM):
    def is_async(self) -> bool:
        return False

    def _initialize(self) -> None:
        self.max_parallel_num = 4
        self.device = getattr(self.vllm_config, 'device', 'cuda')
        self.tensor_parallel_size = getattr(self.vllm_config, 'tensor_parallel_size', 1)

        print(f"Loading vLLM model '{self.model_name}' on device '{self.device}'...")
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
        sampling_params = SamplingParams(
            temperature=getattr(self.generation_config, 'temperature', 0.7),
            max_tokens=getattr(self.generation_config, 'max_tokens', 512),
            top_p=getattr(self.generation_config, 'top_p', None),
            repetition_penalty=getattr(self.generation_config, 'repetition_penalty', None),
        )
        if isinstance(prompts, str):
            prompts = [prompts]
        is_chat = getattr(self.generation_config, 'is_chat', False)
        if is_chat:
            if messages is None:
                messages = [
                    [{"role": "user", "content": prompt}] \
                        if prefix is None else \
                    [{"role": "user", "content": prompt}, {"role": "assistant", "content": prefix}]
                    for prompt in prompts
                ]
            input_texts = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_texts = [text.rstrip(self.tokenizer.eos_token + '\n') for text in input_texts]
        else:
            if messages is not None:
                raise ValueError('`messages` params not support when not in chat mode')
            
    
    async def _async_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None, 
        messages: Optional[List[Dict]] = None,
        prefix: Optional[str] = None
    ) -> str:
        pass


class LLMFactory:
    _models: Dict[ModelType, Type[BaseLLM]] = {
        # ModelType.HF: HuggingFaceLLM,
        ModelType.OPENAI: OpenAILLM,
        # ModelType.VLLM: vLLMLLM,
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
            raise RuntimeError(f"Failed to create LLM instance: {e}")