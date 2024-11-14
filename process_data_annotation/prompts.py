FILL_PROMPT = """Please refer to the given task description and answer, and provide a thought process in the form of step-by-step pseudocode refinement.

A curious user has approached you with a programming question. You should give step-by-step solutions to the user's questions. For each step you can choose one of the following three actions：
<Action 1> Defining Function Structures Using pseudocode
<Action 2> Refine part of the pseudocode
<Action 3> Generate python code from the pseudocode
The pseudocode should be presented in the format: "[Pseudo Start]<PSEUDOCODE>[Pseudo End]". At the final step, provide the complete Python code in this format: "The code is: [Code Start]<CODE>[Code End]." Here, <CODE> should contain a working Python code based on the final pseudocode, and it must be enclosed within Python code block syntax.


**Note!!!**:
1. Please note that the pseudocode should be detailed and provide a step-by-step solution. 
2. Each step should follow the format: "Step x: <Action y>..." (where x is the step number and <Action y> is one of the specified actions).
3. Aim to break down the solution into as many detailed, intermediate steps as possible while ensuring logical coherence between steps and avoiding unnecessary redundancy. The final Python code must be exactly the same as the solution provided in the answer.
4. Strictly follow the format provided in the example.
5. Do not generate content unrelated to the answer or any other explanations.

**In the entire multi-step reasoning process, the pseudocode should smoothly transition from coarse to fine, with each step reflecting the process of refinement. For example, the initial step involves setting up the overall functional blocks of the code (such as using pseudo-comments and TODO: as placeholders), and subsequent steps gradually refine the pseudocode.**

**The format of the Solution must strictly follow the template.**

**Here is an example:**
## Description
You are given a sorted array a of length n. For each i(0<=i<=n-1), you have to make all the elements of the array from index 0 till i equal, using minimum number of operations. In one operation you either increase or decrease the array element by 1.
You have to return a list which contains the minimum number of operations for each i, to accomplish the above task.
Note:
1. 0-based indexing.
2. For each index, you need to consider the same array which was given to you at the start.
Example 1:
Input:
N=4
A={{1,6,9,12}}
Output:
0 5 8 14
Explanation:
For i=0, We do not need to perform any operation, our array will be {{1}}->{{1}}.
And minimum number of operations will be 0. For i=1, We can choose to convert all nthe elements from 0<=j<=i to 4, our array will become {{1,6}}->{{4,4}}. And minimum number of operations will be |1-4|+|6-4|=5. 
For i=2, We can choose to convert all the elements from 0<=j<=i to 6, our array will become {{1,6,9}}->{{6,6,6}} and the minimum number of operations will be |1-6|+|6-6|+|9-6|=8. Similarly, for i=3, we can choose to convert all the elements to 8, {{1,6,9,12}}->{{8,8,8,8}}, and the minimum number of operations will be 14.
Example 2:
Input:
N=7
A={{1,1,1,7,7,10,19}}
Output:
0 0 0 6 12 21 33
Explanation:
Possible operations could be:
{{1}}->{{1}}
{{1,1}}->{{1,1}}
{{1,1,1}}->{{1,1,1}}
{{1,1,1,7}}->{{1,1,1,1}}
{{1,1,1,7,7}}->{{1,1,1,1,1}}
{{1,1,1,7,7,10}}->{{5,5,5,5,5,5}}
{{1,1,1,7,7,10,19}}->{{7,7,7,7,7,7,7}}
Your Task:
You don't need to read input or print anything. Your task is to complete the function optimalArray() which takes N(length of array) and  an array A as input and returns an array of size N  with optimal answer for each index i.
Expected Time Complexity: O(N)
Expected Auxiliary Space: O(1)
Constraints:
1 <= N <= 10^{{6}}
-10^{{5}} <= A[i] <= 10^{{5}}
Sum of N over all test case does not exceed 10^{{6}}.

## Solution
from typing import List

class Solution:
    def optimalArray(self, n: int, a: List[int]) -> List[int]:
        res = [0 for _ in range(n)]
        for i in range(1, n):
            median = a[i // 2]
            res[i] = res[i - 1] + (a[i] - median)
        return res

## Solution with steps
Step 1: <Action 1> Defining Function Structures Using pseudocode
We begin by outlining the structure of the function. We'll need to define a function `optimalArray` that takes the number of elements `n` and a list `a` as input, and returns a list of integers representing the minimum operations for each index `i`. We'll initialize the result list, `res`, and then proceed to compute the necessary operations for each index.

[Pseudo Start]  
```
Function optimalArray(n, a):
    Initialize result list res with n zeros
    Initialize variable ind to 1
    Loop through the array starting from i = 1 to n-1:
        For each i, compute the required operations and update the result
        Increment ind
    Return the result list
[Pseudo End]  
```

Step 2: <Action 2> Refine part of the pseudocode
Let's refine the loop's logic. To compute the minimum number of operations, for each index `i`, we need to make the elements from index `0` to `i` equal. We can use the median of the first `i+1` elements because the median minimizes the sum of absolute deviations. Since the array is sorted, we can find the median by taking the middle element at index `i//2`.

[Pseudo Start]  
```
Function optimalArray(n, a):
    Initialize result list res with n zeros
    Initialize variable ind to 1
    For i = 1 to n-1:
        median = a[i // 2]
        res[ind] = res[i - 1] + (a[i] - median)
        Increment ind
    Return res
[Pseudo End]  
```

Step 3: <Action 2> Refine part of the pseudocode
We need to refine the way we compute the sum of operations. We know that for a sorted array, the total cost of making all elements equal to the median can be computed by summing the absolute differences from the median. We will add the absolute difference between `a[i]` and `median` for each element in the current subarray `a[0]` to `a[i]`.

[Pseudo Start]  
```
Function optimalArray(n, a):
    Initialize result list res with n zeros
    Initialize variable ind to 1
    For i = 1 to n-1:
        median = a[i // 2]
        For j = 0 to i:
            res[i] = res[i-1] + abs(a[j] - median)
    Return res
[Pseudo End]  
```

Step 4: <Action 2> Refine part of the pseudocode
We now optimize the algorithm. The previous step's pseudocode performs a nested loop, which is inefficient. Instead, we will realize that since the array is sorted, the cost of adjusting all elements to the median can be incrementally calculated. At each step, the new element `a[i]` contributes a fixed change relative to the median.

[Pseudo Start]  
```
Function optimalArray(n, a):
    Initialize result list res with n zeros
    For i = 1 to n-1:
        median = a[i // 2]
        res[i] = res[i-1] + (a[i] - median)
    Return res
[Pseudo End]  
```

Step 5: <Action 3> Generate python code from the pseudocode
The pseudocode has now been refined to the point where we can convert it into working Python code. Here's the Python implementation of the `optimalArray` function based on the refined pseudocode.

The code is:
```python
from typing import List

class Solution:
    def optimalArray(self, n: int, a: List[int]) -> List[int]:
        res = [0 for _ in range(n)]
        for i in range(1, n):
            median = a[i // 2]
            res[i] = res[i - 1] + (a[i] - median)
        return res
```

Now, given the following problem description and its corresponding solution, you need to complete the pseudocode refinement solution for this problem:
## Description
{question}

## Solution
{solution}

## Solution with steps
"""

GENERATE_PROMPT = """Please refer to the given task description and provide a thought process in the form of step-by-step pseudocode refinement.

A curious user has approached you with a programming question. You should give step-by-step solutions to the user's questions. For each step you can choose one of the following three actions：
<Action 1> Defining Function Structures Using pseudocode
<Action 2> Refine part of the pseudocode
<Action 3> Generate python code from the pseudocode

## Structure Guidelines:
1. Please note that the pseudocode should be detailed and provide a step-by-step solution. Each step should logically build upon the previous one, moving from high-level structure to detailed implementation.
2. Each step should follow the format: "Step x: <Action y>..." (where x is the step number and <Action y> is one of the specified actions).
3. The pseudocode should be presented in the format: "[Pseudo Start]<PSEUDOCODE>[Pseudo End]".
4. At the final step, provide the complete Python code in this format: "The code is: [Code Start]<CODE>[Code End]." Here, <CODE> should contain a working Python code based on the final pseudocode, and it must be enclosed within Python code block syntax.

## Here are some examples:
{examples}

## Notes
1. Aim to break down the solution into as many detailed, intermediate steps as possible while ensuring logical coherence between steps and avoiding unnecessary redundancy.
2. The Python code solution should match the input and output requirements as described in the question. This means the solution may use terminal I/O for inputs and outputs, or it may require function parameters and return values. Carefully review the question's description to determine the expected code structure, and ensure there are no input/output format errors.
3. Gradually refine each functional part of the pseudocode, breaking down complex operations into manageable steps.
4. Transition to Python code only once all parts of the pseudocode have been fully refined.
5. Strictly follow the format provided in the example.
6. Do not generate content unrelated to the answer or any other explanations.


Now, with the problem description provided below, you need to provide or complete a full, step-by-step solution according to the previous explanations. **If the 'Solution' section is empty, please directly provide a complete, step-by-step solution. If it is not empty, do not repeat or rephrase existing content; simply continue from where it left off to complete the solution.**
### Description
{question}

{solution}"""


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