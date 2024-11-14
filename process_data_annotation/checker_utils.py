import re
import ast
from typing import Optional, Dict, List

from .logger_config import CustomLogger


class CodeSolutionParser:
    def __init__(self):
        pass
        
    def parse_steps(self, text: str) -> list:
        """Parse the solution text into individual steps."""
        # Split by "Step" followed by a number
        step_pattern = r'Step \d+:'
        # Get all starting positions of steps
        step_starts = [m.start() for m in re.finditer(step_pattern, text)]
        
        # Add the end of text position for the last slice
        step_starts.append(len(text))
        
        # Extract each step's content
        steps = []
        for i in range(len(step_starts) - 1):
            step_content = text[step_starts[i]:step_starts[i+1]].strip()
            steps.append(step_content)
            
        return steps
        
    
    def check_final_step(self, steps: List) -> bool:
        """Check if the last step is code generation."""
        if not steps:
            return False
            
        last_step = steps[-1].lower()
        # Check if the last step mentions code generation
        code_indicators = [
            "<Action 3> Generate python code from the pseudocode",
            "the code is:",
        ]
        
        return any(indicator in last_step for indicator in code_indicators)
    
    def extract_code(self, steps: list) -> str:
        """Extract the Python code from the last step."""
        if not steps:
            return None
            
        last_step = steps[-1]
        
        # Find code between triple backticks
        code_pattern = r'```python(.*?)```'
        code_match = re.search(code_pattern, last_step, re.DOTALL)
        
        if code_match:
            return code_match.group(1).strip()
        return None
    
    def extract_outermost_function(self, code: str) -> Optional[Dict]:
        """Extract the outermost function from the code, including class methods."""
        if not code:
            return None
            
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # First try to find module-level function
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    return self._extract_function_info(node)
            
            # If no module-level function found, look for class methods
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    # Look for the first method in the class
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef):
                            # Skip __init__ and other special methods
                            if not class_node.name.startswith('__'):
                                function_info = self._extract_function_info(class_node)
                                function_info['class_name'] = node.name
                                return function_info
                    
        except SyntaxError:
            return None
            
        return None
    
    def _extract_function_info(self, node: ast.FunctionDef) -> Dict:
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
    
    def process_solution(self, text: str) -> Dict:
        """Process the entire solution text and return results."""
        steps = self.parse_steps(text)
        has_code_generation = self.check_final_step(steps)
        code = self.extract_code(steps) if has_code_generation else None
        
        # Extract the outermost function if code exists
        main_function = None
        if code:
            main_function = self.extract_outermost_function(code)
        
        return {
            'total_steps': len(steps),
            'steps': steps,
            'has_code_generation': has_code_generation,
            'final_code': code,
            'main_function': main_function
        }
    
import json
import multiprocessing as mp
import numpy as np
import concurrent.futures
from typing import List, Dict, Union, Optional
from .code_evaluator.testing_util import run_test

TIMEOUT = 10

def check_generation_correctness(
        test_cases: Dict[str, Union[str, List]],
        generation: str,
        timeout: int = TIMEOUT,
        debug: bool = False,
        n_cases: Optional[int] = None,
    ) -> List[bool]:
    """
    Args:
        test_cases (Dict[str, Union[str, List]]): A dictionary containing test cases with inputs and expected outputs.
        generation (str): The generated code to be tested.
        timeout (int, optional): The maximum time allowed for the test execution. Defaults to TIMEOUT.
        debug (bool, optional): If True, prints debug information. Defaults to False.
    Returns:
        List[bool]: A list of booleans indicating the correctness of each test case. If a timeout occurs, returns a list of -1s.
    """
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     try:
    #         future = executor.submit(run_test, test_cases, generation, debug, n_cases)
    #         return future.result()
    #     except concurrent.futures.TimeoutError:
    #         if debug:
    #             print("global timeout")
    #         in_outs = test_cases
    #         return [-1] * len(in_outs["inputs"])
    try:
        return run_test(test_cases, generation, debug, n_cases)
    except Exception as e:
        if debug:
            import traceback
            print(f"Error in running test cases: {traceback.format_exc()}")
        in_outs = test_cases
        return [-2] * len(in_outs["inputs"])
        
def eval_generation(
    generation: str,
    test_cases: Dict[str, Union[str, List]],
    debug: bool = False,
    n_cases: Optional[int] = None,
) -> Dict[str, List]:
    def _normalize_test_result(result):

        if isinstance(result, np.ndarray):
            return result.item(0)
        if isinstance(result, np.bool_):
            return bool(result)
        return result
    try:
        judge_results = check_generation_correctness(test_cases=test_cases, generation=generation, debug=debug, n_cases=n_cases)
        if debug:
            print('[INFO]: Sucessfully run the test cases')
        fixed_judge_results = [_normalize_test_result(result) for result in judge_results]
        if any(res < 1 for res in fixed_judge_results):
            if debug:
                print('\n[INFO]: Code solution failed some test cases')
        return {
            generation: fixed_judge_results
        }
    except Exception as e:
        import traceback
        if debug:
            print(f'[Error]: Error in running test cases: {traceback.format_exc()}')
        return {
            generation: [-2]
        }

def parallel_worker(args):
    generation, test_case, debug, n_cases = args
    result = eval_generation(generation, test_case, debug, n_cases)
    return result

def eval_generations_parallel(
    generations: List[str],
    test_cases: Union[List, Dict[str, Union[str, List]]],
    debug: bool = False,
    n_cases: Optional[int] = None,
    return_binary: bool = True,
) -> Dict[str, Union[int, float]]:
    """Evaluate multiple generations in parallel against a set of test cases.
    Args:
        generations (List[str]): A list of generated strings to be evaluated.
        test_cases (Dict[str, Union[str, List]]): A dictionary containing test cases.
        debug (bool, optional): If True, enables debug mode. Defaults to False.
        return_binary (bool, optional): If True, returns binary results (1 if all test cases pass, 0 otherwise).
                                        If False, returns the proportion of passed test cases. Defaults to True.
    Returns:
        List[Union[int, float]]: A list where each element corresponds to the evaluation result of a generation.
                                 If return_binary is True, the result is binary (1 or 0).
                                 If return_binary is False, the result is a float representing the proportion of passed test cases.
    """
    if not isinstance(test_cases, list):
        test_cases = [test_cases] * len(generations)

    eval_args = [
        (generation, test_case, debug, n_cases) for generation, test_case in zip(generations, test_cases)
    ]

    n_cores = max(1, min(mp.cpu_count() - 2, 8, len(generations)))
    with mp.Pool(n_cores) as pool:
        eval_results = pool.map(parallel_worker, eval_args)
    
    generation_w_status: Dict[str, Union[int, float]] = {}
    for result in eval_results:
        gen, eval_res = list(result.items())[0]
        if return_binary:
            each_generation_passed_cases = int(all(case_res > 0 for case_res in eval_res))
        else:
            each_generation_passed_cases = sum(case_res > 0 for case_res in eval_res) / len(eval_res)
        generation_w_status[gen] = each_generation_passed_cases
        
    return generation_w_status

# def eval_generations_parallel(
#     generations: List[str],
#     test_cases: Union[List, Dict[str, Union[str, List]]],
#     debug: bool = False,
#     n_cases: Optional[int] = None,
#     return_binary: bool = True,
# ) -> Dict[str, Union[int, float]]:
#     if not isinstance(test_cases, list):
#         test_cases = [test_cases] * len(generations)

#     n_workers = max(1, min(4, len(generations)))
#     generation_w_status: Dict[str, Union[int, float]] = {}

#     with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
#         future_to_gen = {
#             executor.submit(eval_generation, gen, test, debug, n_cases): gen 
#             for gen, test in zip(generations, test_cases)
#         }

#         for future in concurrent.futures.as_completed(future_to_gen):
#             gen = future_to_gen[future]
#             try:
#                 result = future.result()
#                 _, eval_res = list(result.items())[0]
#                 if return_binary:
#                     each_generation_passed_cases = int(all(case_res > 0 for case_res in eval_res))
#                 else:
#                     each_generation_passed_cases = sum(case_res > 0 for case_res in eval_res) / len(eval_res)
#                 generation_w_status[gen] = each_generation_passed_cases
#             except Exception as e:
#                 if debug:
#                     import traceback
#                     print(f"Generation evaluation failed: {traceback.format_exc()}")
#                 generation_w_status[gen] = 0

#     return generation_w_status

# def eval_generations_parallel(
#     generations: List[str],
#     test_cases: Union[List, Dict[str, Union[str, List]]],
#     debug: bool = False,
#     n_cases: Optional[int] = None,
#     return_binary: bool = True,
# ) -> Dict[str, Union[int, float]]:
#     if not isinstance(test_cases, list):
#         test_cases = [test_cases] * len(generations)

#     n_workers = max(1, min(16, len(generations)))
#     generation_w_status: Dict[str, Union[int, float]] = {}

#     with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
#         futures = []
#         for gen, test in zip(generations, test_cases):
#             future = executor.submit(eval_generation, gen, test, debug, n_cases)
#             futures.append((future, gen))

#         for future, gen in futures:
#             try:
#                 result = future.result(timeout=30)
#                 _, eval_res = list(result.items())[0]
                
#                 if return_binary:
#                     each_generation_passed_cases = int(all(case_res > 0 for case_res in eval_res))
#                 else:
#                     each_generation_passed_cases = sum(case_res > 0 for case_res in eval_res) / len(eval_res)
                    
#                 generation_w_status[gen] = each_generation_passed_cases
                
#                 if debug:
#                     print(f"Successfully processed generation with result: {each_generation_passed_cases}")
                    
#             except concurrent.futures.TimeoutError:
#                 if debug:
#                     print(f"Timeout processing generation")
#                 generation_w_status[gen] = 0
#             except Exception as e:
#                 if debug:
#                     print(f"Error processing generation: {str(e)}")
#                 generation_w_status[gen] = 0

#     return generation_w_status


def eval_generations_sequential(
    generations: List[str],
    test_cases: Union[List, Dict[str, Union[str, List]]],
    debug: bool = False,
    n_cases: Optional[int] = None,
    return_binary: bool = True,
) -> Dict[str, Union[int, float]]:
    if not isinstance(test_cases, list):
        test_cases = [test_cases] * len(generations)
        
    if debug:
        print(f"[DEBUG] Starting sequential evaluation of {len(generations)} generations")
        
    generation_w_status: Dict[str, Union[int, float]] = {}
    
    for idx, (generation, test_case) in enumerate(zip(generations, test_cases)):
        if debug:
            print(f"[DEBUG] Evaluating generation {idx + 1}/{len(generations)}")
            
        result = eval_generation(generation, test_case, debug, n_cases)
        gen, eval_res = list(result.items())[0]
        
        if return_binary:
            each_generation_passed_cases = int(all(case_res > 0 for case_res in eval_res))
        else:
            each_generation_passed_cases = sum(case_res > 0 for case_res in eval_res) / len(eval_res)
            
        generation_w_status[gen] = each_generation_passed_cases
        
        if debug:
            print(f"[DEBUG] Generation {idx + 1} result: {each_generation_passed_cases}")
            
    return generation_w_status


def postprocess_solutions_and_cases(code_solutions: List[str], test_cases: Dict[str, Union[str, List]]):
    code_solution_parser = CodeSolutionParser()
    code_parsed_infos = [code_solution_parser.process_solution(text=code_solution) for code_solution in code_solutions]
    extended_test_cases = [test_cases] * len(code_solutions)
    code_generations = []
    logger = CustomLogger.get_logger()
    for idx, parsed_info in enumerate(code_parsed_infos):
        if "fn_name" in test_cases and "main_function" in parsed_info \
            and parsed_info["main_function"] is not None and "name" in parsed_info["main_function"]:
            if parsed_info["main_function"]["name"] and parsed_info["main_function"]["name"] != test_cases["fn_name"]:
                extended_test_cases[idx]["fn_name"] = parsed_info["main_function"]["name"]
                logger.warning(f'Code generation {idx+1}\'s main function name is different from the test case\'s fn_name')
        code_generations.append(parsed_info["final_code"])
    
    return code_generations, extended_test_cases