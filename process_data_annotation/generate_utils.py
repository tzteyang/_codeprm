import asyncio
import json
import random
from tqdm import tqdm
from typing import List, Union, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

from .checker_utils import CodeSolutionParser
from .llm_services import BaseLLM, ModelType
from .structure import State
from .prompts import GENERATE_PROMPT


def prompt_prepare(state: State) -> str:
    few_shot_file = './process_data_annotation/data/taco_sampled_examples.json'
    with open(few_shot_file, 'r', encoding='utf-8') as f:
        few_shot_pool = json.loads(f.read())
    few_shot_template = """## Description
{question}

## Solution
{solution}"""
    no_repeat_few_shot = list(filter(lambda x: x["question"] != state.given_problem, few_shot_pool))
    stdio_shot = list(filter(lambda x: x["code_type"] == 'STDIO', no_repeat_few_shot))
    function_shot = list(filter(lambda x: x["code_type"] == 'FUNCTION', no_repeat_few_shot))
    sampled_datas = random.sample(stdio_shot, 1) + random.sample(function_shot, 1)

    few_shot_content = '\n\n'.join([few_shot_template.format(
        question=sampled_data["question"],
        solution=sampled_data["solution_w_steps"]
    ) for sampled_data in sampled_datas])

    def _processed_prefix(prefix):
        return prefix

    prepared_prompt = GENERATE_PROMPT.format(
        examples=few_shot_content,
        question=state.given_problem,
        solution=''
    )

    return prepared_prompt


async def rollout_on_state(
    llm: BaseLLM,
    rollout_num: int,  
    prompt: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    prefix: Optional[str] = None
) -> List[str]:
    
    rollout_responses = []

    if llm.is_async():
        max_requests_num = llm.max_parallel_num
        for i in tqdm(range(0, rollout_num, max_requests_num), leave=True):
            tasks = [
                llm.generate(prompt, messages, prefix)
                for j in range(i, min(i + max_requests_num, rollout_num))
            ]
            responses = await asyncio.gather(*tasks)
            rollout_responses.extend(responses)
    else:
        max_batch_size = llm.max_parallel_num
        for i in tqdm(range(0, rollout_num, max_batch_size), leave=False):
            j = min(i + max_batch_size, rollout_num)
            batch_size = j - i + 1
            if prompt is not None:
                prompts = [prompt] * batch_size
            if messages is not None:
                messages = messages * batch_size
            if prefix is not None:
                prefix = prefix * batch_size
            
            responses = llm.generate(prompts, messages, prefix)

            rollout_responses.extend(responses)

    return rollout_responses


def code_task_postprocess(response: str) -> Dict:
    code_parser = CodeSolutionParser()
    