import re
import glob
import json
import os
import fire
import ast
from tqdm import tqdm
from typing import List, Dict, Any, Union, Optional
from functools import partial

from process_data_annotation.llm_services import (
    LLMFactory, GenerationConfig, vLLMConfig
)
from process_data_annotation.generate_utils import rollout_on_state
from process_data_annotation.checker_utils import eval_generations_parallel


def collect_json_files_with_range(directory_path, start_range=0, end_range=500):
    collected_data = []

    pattern = re.compile(f"train_(\d+)-(\d+)\.json")
    # pattern = re.compile(f"train_(\d+)-(\d+)\.json$")
    
    all_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        match = pattern.search(file_name)
        if match:
            file_start = int(match.group(1))
            file_end = int(match.group(2))
            if file_start >= start_range and file_end <= end_range:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = [json.loads(line) for line in f.readlines()]
                        collected_data.append({
                            'file_name': file_name,
                            'range': (file_start, file_end),
                            'data': data
                        })
                    print(f"read successfully: {file_path}")
                except Exception as e:
                    print(f"when processing file '{file_path}' some error occurred: {e}")
    return collected_data

def extract_code(text_content: str) -> str:
    """Extract the Python code from the last step."""
    if not text_content:
        return None
    # Find code between triple backticks
    code_pattern = r'```python(.*?)```'
    code_match = re.search(code_pattern, text_content, re.DOTALL)
    
    if code_match:
        return code_match.group(1).strip()
    return None

def extract_outermost_function(code: str) -> Optional[Dict]:
    """Extract the outermost function from the code, including class methods."""
    if not code:
        return None
    
    def _extract_function_info(node: ast.FunctionDef) -> Dict:
        """Helper method to extract information from a function node."""
        function_info = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'body': ast.unparse(node)
        }
        
        # Add return type annotation if exists
        if node.returns:
            function_info['return_type'] = ast.unparse(node.returns)
            
        # Add argument type annotations if exist
        arg_types = {}
        for arg in node.args.args:
            if arg.annotation:
                arg_types[arg.arg] = ast.unparse(arg.annotation)
        if arg_types:
            function_info['arg_types'] = arg_types
        
        # Add docstring if exists
        docstring = ast.get_docstring(node)
        if docstring:
            function_info['docstring'] = docstring
            
        return function_info
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # First try to find module-level function
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                return _extract_function_info(node)
        
        # If no module-level function found, look for class methods
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # Look for the first method in the class
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef):
                        # Skip __init__ and other special methods
                        if not class_node.name.startswith('__'):
                            function_info = _extract_function_info(class_node)
                            function_info['class_name'] = node.name
                            return function_info
                
    except SyntaxError:
        return None
        
    return None

def process_solution(generate_solution: str) -> Dict:
    """Process the code solution to extract the main function and final code."""
    final_code = extract_code(generate_solution.strip())
    main_function = None
    if final_code:
        main_function = extract_outermost_function(final_code)

    return {
        'main_function': main_function,
        'final_code': final_code
    }

def postprocess_solutions_and_cases(code_solutions: List[str], test_cases: Dict[str, Union[str, List]]):
    code_parsed_infos = [process_solution(generate_solution=code_solution) for code_solution in code_solutions]
    extended_test_cases = [test_cases] * len(code_solutions)
    code_generations = []

    for idx, parsed_info in enumerate(code_parsed_infos):
        if "fn_name" in test_cases and "main_function" in parsed_info \
            and parsed_info["main_function"] is not None and "name" in parsed_info["main_function"]:
            if parsed_info["main_function"]["name"] and parsed_info["main_function"]["name"] != test_cases["fn_name"]:
                extended_test_cases[idx]["fn_name"] = parsed_info["main_function"]["name"]
                print.warning(f'Code generation {idx+1}\'s main function name is different from the test case\'s fn_name')
        code_generations.append(parsed_info["final_code"])
    
    return code_generations, extended_test_cases

def long_thought_rollout_test(collected_data: List[Dict], output_file: str):
    start_idx = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            start_idx = len(f.readlines())

    # vllm_wrapped_api = LLMFactory.create(
    vllm_wrapped_model = LLMFactory.create(
        model_type='vllm',
        model_name='/root/autodl-tmp/models/Qwen2.5-Coder-32B-Instruct',
        generation_config=GenerationConfig(
            temperature=1.0,
            max_tokens=4096,
            is_chat=True,
            vllm_config=vLLMConfig(
                tensor_parallel_size=1,
                max_model_len=4096,
                # host='localhost',
                # port=8000,
                # api_key='EMPTY'
            )
        )
    )
    # vllm_wrapped_api.max_parallel_num = 8
    vllm_wrapped_model.max_parallel_num = 4
    n_samples = 8
    for idx, data in enumerate(tqdm(collected_data[start_idx:]), start=start_idx):
        question, input_output, completions = data["question"], data["input_output"], data["completions"]
        prompt = data["prompt"]["content"]
        k_samples_rollout_func = partial(
            rollout_on_state, vllm_wrapped_model, rollout_num=n_samples, prompt=prompt
        )
        dummy_rollout_solutions = [None] * n_samples
        all_completions_k_samples_and_tests = []
        for completion in completions:
            code_block_tag = '```python'
            prefix_thought, _ = completion.split(code_block_tag)
            if not prefix_thought:
                all_completions_k_samples_and_tests.append(
                    postprocess_solutions_and_cases(dummy_rollout_solutions, input_output)
                ) # no prefix thought
                continue
            prefix_thought += code_block_tag + '\n'
            # breakpoint()
            rollout_solutions = k_samples_rollout_func(
                prefix=prefix_thought
            )
            breakpoint()
            rollout_solutions = [code_block_tag + '\n' + sol for sol in rollout_solutions]
            all_completions_k_samples_and_tests.append(
                postprocess_solutions_and_cases(rollout_solutions, input_output)
            )

        completions_k_samples = []
        for (code_generations, updated_test_cases) in all_completions_k_samples_and_tests:
            results = eval_generations_parallel(code_generations, updated_test_cases, debug=False, n_cases=100)
            _curr_k_samples = []
            for pass_status in results.values():
                _curr_k_samples.append(pass_status)
            completions_k_samples.append(_curr_k_samples)
            breakpoint()
        
        tested_data = {
            "question": question,
            "input_output": input_output,
            "completions": completions,
            "completions_k_samples": completions_k_samples
        }

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(tested_data, ensure_ascii=False) + '\n')


def main(search_dir: str):
    collected_data = collect_json_files_with_range(search_dir, 0, 1000)
    # breakpoint()
    for data_block in collected_data:
        file_name, file_extension = os.path.splitext(data_block['file_name'])
        output_file = os.path.join(search_dir, f"{file_name}_rollout_test{file_extension}")

        long_thought_rollout_test(data_block['data'], output_file)


if __name__ == '__main__':
    fire.Fire(main)