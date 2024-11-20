import torch
import warnings
from dataclasses import dataclass, field
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Union, Dict, Literal, TypeAlias


CODEPRM_PROMPT = """Please refer to the given task description and provide a thought process in the form of step-by-step pseudocode refinement.

A curious user has approached you with a programming question. You should give step-by-step solutions to the user's questions. For each step you can choose one of the following three actions：
<Action 1> Defining Function Structures Using pseudocode
<Action 2> Refine part of the pseudocode
<Action 3> Generate python code from the pseudocode

## Structure Guidelines:
1. Please note that the pseudocode should be detailed and provide a step-by-step solution. Each step should logically build upon the previous one, moving from high-level structure to detailed implementation.
2. Each step should follow the format: "Step x: <Action y>..." (where x is the step number and <Action y> is one of the specified actions).
3. The pseudocode should be presented in the format: "[Pseudo Start]<PSEUDOCODE>[Pseudo End]".
4. At the final step, provide the complete Python code in this format: "The code is: [Code Start]<CODE>[Code End]." Here, <CODE> should contain a working Python code based on the final pseudocode, and it must be enclosed within Python code block syntax.

## Notes
1. Aim to break down the solution into as many detailed, intermediate steps as possible while ensuring logical coherence between steps and avoiding unnecessary redundancy.
2. The Python code solution should match the input and output requirements as described in the question. This means the solution may use terminal I/O for inputs and outputs, or it may require function parameters and return values. Carefully review the question's description to determine the expected code structure, and ensure there are no input/output format errors.
3. Gradually refine each functional part of the pseudocode, breaking down complex operations into manageable steps.
4. Transition to Python code only once all parts of the pseudocode have been fully refined.
6. Do not generate content unrelated to the answer or any other explanations.

Now, with the problem description provided below, you need to provide or complete a full, step-by-step solution according to the previous explanations. **If the 'Solution' section is empty, please directly provide a complete, step-by-step solution. If it is not empty, do not repeat or rephrase existing content; simply continue from where it left off to complete the solution.**
### Description
{question}

### Solution
"""


@dataclass
class StepTokensForLM:
    step_tag: str = field(
        default=' Rating', 
        metadata={'help': 'The tag that indicates the end of a step/action'}
    )
    good_token: str = field(
        default=' +', 
        metadata={'help': 'The token that indicates a positive action'}
    )
    bad_token: str = field(
        default=' -', 
        metadata={'help': 'The token that indicates a negative action'}
    )

@dataclass
class RewardStrategy(Enum):
    """
    Enum class for the tokenized format of the text.
    """
    TOKEN_LOGITS = 'token_logits'
    VALUE_HEAD = 'value_head'


PromptType: TypeAlias = Union[str, List[str]]
PrefixesType: TypeAlias = List[PromptType]

@torch.no_grad()
def get_process_rewards(model: AutoModelForCausalLM,
                         tokenizer: AutoTokenizer,
                         prompts: PromptType,
                         completed_processes: PrefixesType,
                         tokenized_format: Optional[Literal['completion', 'chat_completion']],
                         reward_strategy: Optional[RewardStrategy] = RewardStrategy.TOKEN_LOGITS.value,
                         ) -> List[float]:
    
    if isinstance(prompts, str):
        if isinstance(completed_processes[0], list):
            raise ValueError("The `completed_prefixes` argument must be a list of strings if `prompts` is a string.")
        prompts = [prompts]
        completed_processes = [completed_processes]
    if len(prompts) != len(completed_processes):
        raise ValueError("The number of prompts must match the number of completed prefixes in order.")

    if reward_strategy is RewardStrategy.TOKEN_LOGITS.value:
        tokenized_ids = tokenizer(
            [StepTokensForLM.step_tag, StepTokensForLM.good_token, StepTokensForLM.bad_token]
        )["input_ids"]
        if any(len(tokenized_id) != 1 for tokenized_id in tokenized_ids):
            raise ValueError("The tokens `step_tag`, `good_token`, `bad_token` used in class `StepTokensForLM` must be single tokens.")
        step_token_id, good_token_id, bad_token_id = [ids[0] for ids in tokenized_ids]

        if tokenized_format == 'completion':
            raise NotImplementedError("Token Logits reward strategy is not implemented for completion tokenized format.")
        elif tokenized_format == 'chat_completion':
            input_texts = []
            for prompt, processes in zip(prompts, completed_processes):
                step_tag_inserted_process = ''
                for process in processes:
                    step_tag_inserted_process += process + StepTokensForLM.step_tag
                    
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": step_tag_inserted_process}]
                chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

                end_token_of_one_turn = '<|im_end|>\n'
                chat_template = chat_template.removesuffix(end_token_of_one_turn)
                input_texts.append(chat_template)
            # breakpoint()
            model_inputs = tokenizer(input_texts, return_tensors='pt', padding=True).to(model.device)
            step_tag_positions = (model_inputs["input_ids"] == step_token_id).nonzero()
            model_inputs["attention_mask"][step_tag_positions[:, 0], step_tag_positions[:, 1]] = 0

            outputs = model(**model_inputs)
            logits = outputs.logits
            logits = logits[..., [good_token_id, bad_token_id]]
            scores = logits[step_tag_positions[:, 0], step_tag_positions[:, 1] - 1, :].softmax(dim=-1) # [bs, 2(' +', ' -')]
            
            return scores
        else:
            raise ValueError("The `tokenized_format` argument must be either 'completion' or 'chat_completion'.")
    

if __name__ == '__main__':
    model_path = '/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct-PRM'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    problem = 'Given some positive integers, I wish to print the integers such that all take up the same width by adding a minimum number of leading zeroes. No leading zeroes shall be added to the largest integer.\n\nFor example, given `1, 23, 2, 17, 102`, I wish to print out these numbers as follows:\n\n```python\n001\n023\n002\n017\n102\n```\n\nWrite a function `print_nums(n1, n2, n3, ...)` that takes a variable number of arguments and returns the string to be printed out.'
    prompt = CODEPRM_PROMPT.format(question=problem)
    examples = [
        "Step 1: <Action 1> Defining Function Structures Using pseudocode\nWe start by defining the structure of our solution. We need a function `reorder` that takes two integers `N` and `M`, and returns a numpy array with two sub-arrays. Each sub-array will contain numbers in the specified ranges and will be rotated `M` times.\n\n[Pseudo Start]\n```\nFunction reorder(N, M):\n    Calculate half of N\n    Create the first sub-array with numbers in the range [0, N/2)\n    Create the second sub-array with numbers in the range [N/2, N)\n    Rotate the first sub-array M times\n    Rotate the second sub-array M times\n    Combine the two sub-arrays into a numpy array\n    Return the combined numpy array\n[Pseudo End]\n```"
    ]

    get_process_rewards(model, tokenizer, prompt, examples, 'chat_completion', 'token_logits')