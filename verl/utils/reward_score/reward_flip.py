"""
Reward function for image flip tool use training.
Compares the answer in <boxed></boxed> tags with the expected ground truth.
Returns 1.0 for correct answer, 0.0 for incorrect.
"""

import re
import logging
import sys

logger = logging.getLogger("reward_flip")
logger.setLevel(logging.ERROR)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.ERROR)
    h.setFormatter(logging.Formatter("[FLIP_REWARD] %(message)s"))
    logger.addHandler(h)


def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer from <boxed>...</boxed> tags.
    
    Args:
        text: The text to search for boxed answer.
        
    Returns:
        The extracted answer string or None if not found.
    """
    if not text:
        return None
    
    # Try multiple patterns for boxed answers
    patterns = [
        r'<boxed>(.*?)</boxed>',  # <boxed>answer</boxed>
        r'\\boxed\{(.*?)\}',       # \boxed{answer}
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            # Return the last match (most likely the final answer)
            return matches[-1].strip().lower()
    
    return None


def extract_tool_calls_from_text(text: str) -> list:
    """
    Extract properly formatted tool calls using <tool_call>...</tool_call> tags.
    Returns list of parsed tool call dicts, or empty list if none found.
    """
    import json
    if not text:
        return []
    
    tool_calls = []
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                tool_calls.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return tool_calls


def compute_score(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """Score a flip task output.

    Returns a float in [0, 1]. solution_str is the model output; ground_truth
    is the expected answer. extra_info may contain tool execution metadata.
    """
    solution = str(solution_str or "").strip()
    
    # Check for tool execution errors from agent loop
    tool_execution_errors = 0
    if extra_info:
        tool_execution_errors = extra_info.get("tool_execution_errors", 0)
    tool_execution_failed = tool_execution_errors > 0
    
    # Check if model used proper tool call format
    tool_calls = extract_tool_calls_from_text(solution)
    used_tool = len(tool_calls) > 0
    
    # Extract answer from solution
    predicted_answer = extract_boxed_answer(solution)
    
    # Extract answer from ground truth (in case it's in boxed format)
    expected_answer = extract_boxed_answer(ground_truth)
    if expected_answer is None:
        # Ground truth might be a plain string
        expected_answer = str(ground_truth).strip().lower()
    
    # Compute answer correctness
    answer_correct = predicted_answer is not None and predicted_answer == expected_answer
    
    # Determine final score
    if tool_execution_failed:
        # Tool call failed to execute - zero reward to train model to avoid malformed calls
        score = 0.0
        logger.error(f"Tool execution failed! Predicted: {predicted_answer}, Expected: {expected_answer}, Errors: {tool_execution_errors}")
    elif not used_tool:
        # No tool call detected - zero reward (model must learn to use the tool)
        score = 0.0
        logger.error(f"No tool call found! Predicted: {predicted_answer}, Expected: {expected_answer}")
    elif answer_correct:
        # Tool used correctly and answer is correct
        score = 1.0
        logger.error(f"Correct! Predicted: {predicted_answer}, Expected: {expected_answer}")
    else:
        # Tool used but answer wrong - small reward for trying
        score = 0.1
        logger.error(f"Wrong answer. Predicted: {predicted_answer}, Expected: {expected_answer}")
    
    return score
