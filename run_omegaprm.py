import json
import random
import os
import yaml
import asyncio
import logging
import re
from typing import (
    Dict, Union, List
)
from tqdm import tqdm
from argparse import ArgumentParser
from process_data_annotation.omegaprm import CodeOmegaPRM
from process_data_annotation.llm_services import (
    BaseLLM, GenerationConfig, LLMFactory, vLLMConfig
)
from process_data_annotation.structure import State
from process_data_annotation.logger_config import CustomLogger
from process_data_annotation.hparams import (
    DataParams, ModelParams, SearchAlgorithmParams
)
from process_data_annotation.generate_utils import (
    rollout_on_state, prompt_prepare
)
from process_data_annotation.checker_utils import (
    eval_generations_parallel, postprocess_solutions_and_cases, eval_generations_sequential
)

logger = None

def load_run_examples(from_path: str):
    # random.seed(42)
    with open(from_path, 'r', encoding='utf-8') as f:
        run_examples = json.load(f)
    # random.shuffle(run_examples)
    difficulty_levels = ["EASY", "MEDIUM"]
    run_examples = list(filter(lambda x: x['difficulty'] in difficulty_levels, run_examples))
    return run_examples


def should_further_process(LM: BaseLLM, problem: Dict[str, Union[str, Dict]]) -> bool:
    question, test_cases = problem['question'], problem['test_cases']
    temp_state = State(solution_prefix='### Solution')
    temp_state.given_problem = question
    code_solutions = rollout_on_state(
        LM,
        rollout_num=32,
        prompt=prompt_prepare(temp_state),
        prefix=temp_state.solution_prefix
    )
    global logger
    # with open('./process_data_annotation/data/code_solutions_examples.json', 'r', encoding='utf-8') as f:
        # f.write(json.dumps(code_solutions, indent=4, ensure_ascii=False))
        # code_solutions = json.load(f)
    code_generations, extended_test_cases = postprocess_solutions_and_cases(code_solutions, test_cases)
    generation_w_status = eval_generations_parallel(code_generations, extended_test_cases, debug=False, n_cases=100)
    has_passed, has_failed = any(int(res) == 1 for res in generation_w_status.values()), any(int(res) == 0 for res in generation_w_status.values())
    init_test_pass_rate = sum(int(res) == 1 for res in generation_w_status.values()) / len(generation_w_status)
    logger.info(f'Initial test pass rate: {init_test_pass_rate}')

    return has_passed and has_failed

def save_collected_data(collected_data: Dict, index: int, output_dir: str):
    global logger
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"problem_{index}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(collected_data, indent=4, ensure_ascii=False))
    logger.info(f'Saved processed data to {filename}')


def exist_problem_count(data_params: DataParams) -> int:
    os.makedirs(data_params.output_dir, exist_ok=True)
    existing_files = os.listdir(data_params.output_dir)
    existing_indices = []
    for filename in existing_files:
        match = re.match(r'problem_(\d+)\.json', filename)
        if match:
            idx = int(match.group(1))
            existing_indices.append(idx)
    if existing_indices:
        starting_idx = max(existing_indices) + 1
    else:
        starting_idx = 0
    return starting_idx


def create_generation_config(model_params: ModelParams) -> GenerationConfig:
    vllm_config = None
    if model_params.model_type.lower() in ['vllm', "vllm_server"]:
        if not model_params.vllm_config:
            raise ValueError('VLLM config is required for VLLM model type')
        vllm_config = vLLMConfig(**model_params.vllm_config)
        
    return GenerationConfig(
        temperature=model_params.temperature,
        max_completion_tokens=model_params.max_completion_tokens,
        max_tokens=model_params.max_tokens,
        is_chat=model_params.is_chat,
        vllm_config=vllm_config
    )


def run_exp(args):
    with open(args.config_file, 'r') as f:
        run_config = yaml.safe_load(f)

    data_params = DataParams.from_dict(run_config['data_params'])
    model_params = ModelParams.from_dict(run_config['model_params'])
    search_algorithm_params = SearchAlgorithmParams.from_dict(run_config['search_algorithm_params'])
    print(data_params, model_params, search_algorithm_params, sep='\n')
    run_examples = load_run_examples(data_params.data_file)

    generation_config = create_generation_config(model_params)

    llm: BaseLLM = LLMFactory.create(
        model_type=model_params.model_type,
        model_name=model_params.model_name,
        generation_config=generation_config
    )

    code_omega_prm = CodeOmegaPRM(
        LM=llm,
        c_puct=search_algorithm_params.c_puct,
        alpha=search_algorithm_params.alpha,
        beta=search_algorithm_params.beta,
        L=search_algorithm_params.length_scale,
        k=search_algorithm_params.num_rollouts,
        N=search_algorithm_params.max_search_count,
        rollout_budget=search_algorithm_params.rollout_budget,
    )

    start_idx = exist_problem_count(data_params)
    print('*' * 50)
    print(f'\nRestart from problem {start_idx}\n')
    print('*' * 50)
    for idx, problem in enumerate(tqdm(run_examples[start_idx:]), start=start_idx):
        CustomLogger.set_log_file(os.path.join(data_params.logs_dir, f'problem_{idx}.log'))
        global logger
        logger = CustomLogger.get_logger()
        if not should_further_process(llm, problem):
            logger.info('*' * 10 + f' Skip further processing for problem {idx} ' + '*' * 10)
            continue
        logger.info('*' * 10 + f' Processing problem {idx} ' + '*' * 10)

        collected_data = {"task_id": problem["task_id"], "question": problem["question"]}
        collected_data.update(code_omega_prm.run(
            problem=problem["question"],
            test_cases=problem["test_cases"]
        ))
        save_collected_data(collected_data, idx, data_params.output_dir)
        

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Path to the config file')
    
    args = parser.parse_args()
    
    run_exp(args)