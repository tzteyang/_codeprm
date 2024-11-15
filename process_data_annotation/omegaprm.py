import math
import re
import json
import asyncio
from typing import List, Tuple, Dict, Any, Optional, Union

from .structure import (
    State, SearchTree, CandidatePool
)
from .llm_services import BaseLLM
from .generate_utils import rollout_on_state, prompt_prepare
from .logger_config import CustomLogger
from .checker_utils import (
    postprocess_solutions_and_cases, eval_generations_parallel, CodeSolutionParser
)

def separate_code_steps(solution: str) -> List[str]:
    code_parser = CodeSolutionParser()
    reason_steps = code_parser.process_solution(solution)["steps"]
    return reason_steps


# Define the OmegaPRM algorithm
class CodeOmegaPRM:
    def __init__(self, LM: BaseLLM,  c_puct: float, alpha: float, beta: float, L: int, k: int, N: int,
                 rollout_budget: int):
        """
        Initialize the OmegaPRM algorithm.

        Parameters:
        - LM (LanguageModel): The language model instance.
        - expected_answer (str): The expected answer for correctness checking.
        - c_puct (float): Exploration constant.
        - alpha (float): Weight for MC(s).
        - beta (float): Length penalty.
        - L (int): Maximum solution length.
        - k (int): Number of rollouts for Monte Carlo estimation.
        - N (int): Maximum search count.
        """
        self.LM = LM  # Language Model
        self.expected_answer = None
        self.c_puct = c_puct
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.k = k
        self.N = N
        self.rollout_budget = rollout_budget

        self.code_test_cases = None

        self.T = SearchTree()
        self.C = CandidatePool()

        self.n = 0
        self.total_rollouts = 0
        self.collected_data = []

    def reset(self):
        """Reset internal state variables to prepare for a fresh run."""
        self.code_test_cases = None
        self.T = SearchTree()  # Reset search tree
        self.C = CandidatePool()  # Reset candidate pool
        self.n = 0
        self.total_rollouts = 0
        self.collected_data = []  # Clear collected data

    def run(self, problem: str, test_cases: Dict[str, Union[List, List[List]]]) -> List:
        """
        Execute the OmegaPRM algorithm.

        Parameters:
        - question (str): The question to generate solutions for.

        Returns:
        - Collected data: List of dictionaries.
        """
        self.reset()
        
        logger = CustomLogger.get_logger()
        logger.info(f"Running OmegaPRM for question: {problem}\n")
        # Initialization
        initial_state = State(solution_prefix='### Solution\n', parent=None)
        initial_state.set_given_problem(problem)

        self.code_test_cases = test_cases
        self.T.root = initial_state
        self.T.add_state(initial_state)
        self.n = 0

        # Monte Carlo Estimation for initial_state
        self.monte_carlo_estimation(initial_state)

        # Main loop
        while self.n < self.N and self.total_rollouts < self.rollout_budget and not self.C.is_empty():
            # Selection Phase
            selected_state, selected_rollout = self.selection_phase()
            if selected_state is None or selected_rollout is None:
                logger.warning("No more candidates to explore. Terminating search.")
                break

            # Log the selected rollout
            state_id = self.T.nodes.index(selected_state)
            logger.debug("=" * 15 + f"Selection Phase: Selected rollout from State ID {state_id}" + '=' * 15)
            logger.debug(f"Selected state MC: {selected_state.MC}")
            logger.debug(f"Selected state N: {selected_state.N}")
            logger.debug(f"Selected state Q_and_U: {self.compute_selection_score(selected_state, selected_rollout)}")
            logger.debug(f'Candidate pool valid item nums: {len(self.C)}')

            logger.info(f"Selected Rollout:\n{selected_rollout}\n")

            if hasattr(selected_state, 'has_been_selected'):
                selected_state.has_been_selected.add(selected_rollout)

            # Proceed to expansion phase with binary search
            self.expansion_phase_binary_search(selected_state, selected_rollout)

            # Maintenance Phase
            self.maintenance_phase(selected_state)

            # Increment search count
            selected_state.N += 1
            self.n += 1
            logger.debug('#' * 30)
            logger.info(f'End of Iteration {self.n}')
            logger.debug('#' * 30)

        steps_data = self.collect_solution_prefixes()
        tree_data = self.collect_tree_structure()
        return {
            "steps_data": steps_data,
            "tree_data": tree_data
        }

    def is_termianl_state(self, state: State) -> bool:
        """
        Determines if the current state is a terminal state in the solution generation process.

        A terminal state is reached when the solution contains generated code.

        Args:
            state (State): The current state object containing solution information.

        Returns:
            bool: True if the state is terminal (has generated code), False otherwise.
        """
        solution_parser = CodeSolutionParser()
        solution_info = solution_parser.process_solution(state.solution_prefix)
        is_terminal = solution_info["has_code_generation"]
        return is_terminal


    def monte_carlo_estimation(self, state: State):
        """
        Perform Monte Carlo estimation for state by generating k rollouts
        and computing MC(s) = c / k, where c is the number of correct rollouts.
        """
        logger = CustomLogger.get_logger()
        if self.is_termianl_state(state):
            logger.warning(f"State ID {self.T.nodes.index(state)} is a terminal state. No further rollouts will be added.")
            state.MC = 0.0
            return # Skip Monte Carlo estimation for terminal states

        c = 0  # Correct rollouts count
        incorrect_rollouts = []
        correct_rollouts = []

        # asynchronous parallel rollout sampling

        logger.debug('*' * 15 + ' Solution prefix for rollout ' + '*' * 15)
        logger.info(state.solution_prefix)
        k_rollout_solutions = rollout_on_state(
            llm=self.LM,
            rollout_num=self.k,
            prompt=prompt_prepare(state),
            prefix=state.solution_prefix
        )
        logger.debug('*' * 15 + ' One of rollout solutions ' + '*' * 15)
        logger.info(k_rollout_solutions[0])
        # breakpoint()
        self.total_rollouts += self.k
        
        # Handle cases where subsequent rollouts may duplicate content from the current state's solution_prefix
        prefix_duplicate_k_solution_rollouts_steps = None
        if state.solution_prefix:
            solution_parser = CodeSolutionParser()
            solution_prefix_steps = solution_parser.process_solution(state.solution_prefix)["steps"]
            k_solution_rollouts_steps = [solution_parser.process_solution(rollout_solution)["steps"] for rollout_solution in k_rollout_solutions]
            pattern = r'Step (\d+):'
            prefix_duplicate_k_solution_rollouts_steps = []
            solution_prefix_steps_count = len(solution_prefix_steps)

            for rollout_steps in k_solution_rollouts_steps:
                _temp_steps = []
                for step in rollout_steps:
                    step_id = re.findall(pattern, step)
                    if step_id:
                        if int(step_id[0]) > solution_prefix_steps_count:
                            _temp_steps.append(step)
                    else:
                        _temp_steps.append(step)
                prefix_duplicate_k_solution_rollouts_steps.append('\n\n'.join(_temp_steps))
            assert len(prefix_duplicate_k_solution_rollouts_steps) == len(k_rollout_solutions)
            k_rollout_solutions = ['\n\n'.join([state.solution_prefix, rollout]) for rollout in prefix_duplicate_k_solution_rollouts_steps]

        # check each rollout's correctness
        code_generations, extended_test_cases = postprocess_solutions_and_cases(k_rollout_solutions, self.code_test_cases)
        generation_w_status = eval_generations_parallel(code_generations, extended_test_cases, debug=False, n_cases=100)
        for solution, (code_gen, code_status) in zip(
            k_rollout_solutions if prefix_duplicate_k_solution_rollouts_steps is None else prefix_duplicate_k_solution_rollouts_steps, 
            generation_w_status.items()
        ):
            if code_status == 1:
                c += 1
                correct_rollouts.append(solution)
            else:
                incorrect_rollouts.append(solution)
                state.add_incorrect_rollout(solution)

        # Update total rollouts and correct rollouts
        state.total_rollouts += self.k
        state.correct_rollouts += c
        state.MC = state.correct_rollouts / state.total_rollouts if state.total_rollouts > 0 else 0
        logger.debug('=' * 15 + ' End of Monte Carlo Estimation ' + '=' * 15)
        logger.info(f"Current rollout costs: {self.total_rollouts}, Total rollout budget: {self.rollout_budget}")
        logger.info(f"Monte Carlo Estimation for State ID {self.T.nodes.index(state)}: MC = {state.MC:.2f}")
        logger.info(f'\tTotal Rollouts = {state.total_rollouts}')
        logger.info(f'\tCorrect Rollouts = {state.correct_rollouts}')

        if state.MC == 1.0:
            # Add all correct rollouts to the tree as new states
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
        elif state.MC == 0.0:
            # State is incorrect; no further action
            return
        else:
            # 0 < MC(s) < 1.0
            # Add correct rollouts to the tree
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
            # Add incorrect rollouts to candidate pool with updated priorities
            unique_incorrect_rollouts = set(incorrect_rollouts)
            logger.debug('\n' + '=' * 10 + f" Candidate Pool Insert Incorrect Rollouts for State ID {self.T.nodes.index(state)}: " + '=' * 10)
            print(f"Total Incorrect Rollouts: {len(incorrect_rollouts)}")
            print(f"Unique Incorrect Rollouts: {len(unique_incorrect_rollouts)}")
            for rollout in incorrect_rollouts:
                priority = self.compute_selection_score(state, rollout)
                self.C.add_or_update(state, rollout, priority)


    def compute_Q(self, state: State, rollout: str) -> float:
        """
        Compute Q(s, r) = alpha^{1 - MC(s)} * beta^{len(r)/L}, where len(r) is based on word count.
        """
        # Count words in the rollout
        # word_count = len(rollout.split())
        word_count = self.LM.num_tokens_from_string(string=rollout)
        length_penalty = word_count / self.L
        Q_value = (self.alpha ** (1 - state.MC)) * (self.beta ** length_penalty)
        return Q_value

    def compute_U(self, state: State) -> float:
        """
        Compute U(s) = c_puct * sqrt(sum_{s'} N(s')) / (1 + N(s))
        """
        N_total = sum(s.N for s in self.T.nodes)
        if N_total == 0:
            N_total = 1  # Prevent division by zero
        U_s = self.c_puct * (math.sqrt(N_total)) / (1 + state.N)
        return U_s

    def compute_selection_score(self, state: State, rollout: str) -> float:
        """
        Compute selection score: Score(s, r) = Q(s, r) + U(s)
        """
        Q_s_r = self.compute_Q(state, rollout)
        U_s = self.compute_U(state)
        score = Q_s_r + U_s
        return score

    def selection_phase(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Select (state, rollout) with the highest score from candidate pool C.
        """
        selected_state, selected_rollout = self.C.pop()
        return selected_state, selected_rollout

    def add_correct_rollout_to_tree(self, parent_state: State, rollout: str):
        """
        Add the correct rollout to the tree as a child of parent_state.
        """
        new_solution_prefix = (parent_state.solution_prefix + '\n\n' + rollout).strip() if parent_state.solution_prefix else rollout
        new_state = State(solution_prefix=new_solution_prefix, parent=parent_state)
        new_state.MC = 1.0  # Since the rollout is correct
        new_state.total_rollouts = 0
        new_state.correct_rollouts = 0
        self.T.add_state(new_state)
        parent_state.children.append(new_state)  # Add to parent's children

    def expansion_phase_binary_search(self, parent_state: State, rollout: str):
        """
        Expansion phase that adds the rollout as a new state and performs Monte Carlo estimation
        using Binary Search to efficiently find the correct rollout.

        Parameters:
        - parent_state (State): The state from which the rollout was selected.
        - rollout (str): The rollout string that was selected and is incorrect.
        """
        # Separate the rollout into individual steps
        # steps = separate_steps(rollout, mode='split')
        steps = separate_code_steps(rollout)
        logger = CustomLogger.get_logger()
        # Perform binary search to find incorrect steps

        self.binary_search_incorrect_step(parent_state, steps, 0, len(steps) - 1)
        logger.debug(f'State ID {self.T.nodes.index(parent_state)} finished binary search for incorrect steps.')

    def binary_search_incorrect_step(self, s_ast: State, steps: List[str], left: int, right: int):
        """
        Recursively perform binary search to find all incorrect steps in the rollout.

        Parameters:
        - s_ast (State): The selected parent state.
        - steps (List[str]): The rollout steps as a list.
        - left (int): Left index of the current search interval.
        - right (int): Right index of the current search interval.
        """
        if left > right:
            return
        logger = CustomLogger.get_logger()

        mid = (left + right) // 2
        # Create prefix solution up to mid
        new_steps = steps[left:mid + 1]
        # prefix_solution = (s_ast.solution_prefix + '\n\n' + '\n\n'.join(new_steps)).strip() if s_ast.solution_prefix else '\n\n'.join(new_steps).strip()
        # prefix_solution = (s_ast.solution_prefix + '\n\n' + separate_steps(new_steps, mode='join')).strip() if s_ast.solution_prefix else separate_steps(new_steps, mode='join').strip()
        if new_steps:
            prefix_solution = s_ast.solution_prefix + '\n\n' + '\n\n'.join(new_steps).strip()
        else:
            prefix_solution = s_ast.solution_prefix
        # Create new state s_new
        s_new = State(solution_prefix=prefix_solution.strip(), parent=s_ast)
        self.T.add_state(s_new)
        s_ast.children.append(s_new)
        state_id_new = len(self.T.nodes) - 1

        # Perform Monte Carlo estimation for s_new
        self.monte_carlo_estimation(s_new)

        if s_new.MC == 0:
            # Found incorrect step; continue searching in the left half to find earlier incorrect steps
            logger.debug('>>' * 10 + ' Binary Search Info ' + '<<' * 10) 
            logger.info(f'State ID {state_id_new} has MC == 0. Incorrect Step {mid + 1} found. Searching earlier steps.')
            self.binary_search_incorrect_step(s_ast, steps, left, mid - 1)
        else:
            # Steps up to mid are correct; continue searching in the right half
            logger.debug('>>' * 10 + ' Binary Search Info ' + '<<' * 10)
            logger.info(f"State ID {state_id_new} has MC == {s_new.MC:.2f}. Steps up to Step {mid + 1} are correct. Searching later steps.")
            self.binary_search_incorrect_step(s_new, steps, mid + 1, right)

    def maintenance_phase(self, state: State):
        """
        Update statistics and candidate pool for all incorrect rollouts associated with the state.

        Parameters:
        - state (State): The state whose incorrect rollouts need to be updated.
        """
        logger = CustomLogger.get_logger()
        solution_parser = CodeSolutionParser()

        unique_incorrect_rollouts = set(state.incorrect_rollouts)
        logger.debug(f"Candidate Pool Insert Incorrect Rollouts for State ID {self.T.nodes.index(state)}:")
        logger.info(f"Total Incorrect Rollouts: {len(state.incorrect_rollouts)}")
        logger.info(f"Unique Incorrect Rollouts: {len(unique_incorrect_rollouts)}")
        # Iterate through all incorrect rollouts of the state
        for rollout in state.incorrect_rollouts:
            rollout_info = solution_parser.process_solution(rollout)
            # skip rollouts which is the last step of the solution
            if rollout_info["total_steps"] <= 1 and rollout_info["has_code_generation"]:
                continue
            # Since we've already determined these rollouts are incorrect, no need to re-evaluate correctness
            priority = self.compute_selection_score(state, rollout)
            # Update the candidate pool with the new priority
            self.C.add_or_update(state, rollout, priority)

        logger.info("Maintenance Phase Completed.")

    def collect_solution_prefixes(self) -> List[Dict[str, Any]]:
        """
        Collect all solution prefixes and their corresponding MC values from the search tree.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing solution prefixes and their MC values.
        """
        collected_data = []
        for node in self.T.nodes:
            solution_prefix = node.solution_prefix
            mc_value = node.MC
            collected_data.append({
                "node_id": self.T.nodes.index(node),
                "solution_prefix": solution_prefix,
                "mc_value": mc_value
            })
        return collected_data

    def collect_tree_structure(self) -> Dict[str, Any]:
        """
        Collect the tree structure starting from the root.

        Returns:
            Dict[str, Any]: A nested dictionary representing the tree structure.
        """
        if self.T.root:
            tree_data = self.T.root.get_text_with_labels(self.T.nodes)
            return tree_data
        return {}


# Example usage
if __name__ == "__main__":
    ...