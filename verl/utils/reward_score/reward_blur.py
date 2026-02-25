# reward_blur.py
"""
Reward function for image blur tool use training.
Compares the answer in <boxed></boxed> tags with the expected ground truth.
Returns 1.0 for correct answer, 0.0 for incorrect.
"""

import re
import logging
import sys

logger = logging.getLogger("reward_blur")
logger.setLevel(logging.ERROR)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.ERROR)
    h.setFormatter(logging.Formatter("[BLUR_REWARD] %(message)s"))
    logger.addHandler(h)


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the answer from <boxed>...</boxed> tags

    """
    if not text:
        return None
    
    # Try multiple patterns for boxed answers
    patterns = [
        r'<boxed>(.*?)</boxed>', 
        r'\\boxed\{(.*?)\}',       
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip().lower()
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    """
    if answer is None:
        return ""
    
    answer = answer.lower().strip()
    answer = re.sub(r'[.,!?;:\'"]+', '', answer)
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    
    return answer


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
    """Score a blur task output.

    Returns a float in [0, 1]. solution_str is the model output; ground_truth
    is the expected answer. extra_info may contain tool execution metadata.
    """
    if not solution_str or not ground_truth:
        return 0.0
    
    solution = str(solution_str).strip()
    
    # Check for tool execution errors from agent loop
    tool_execution_errors = 0
    if extra_info:
        tool_execution_errors = extra_info.get("tool_execution_errors", 0)
    tool_execution_failed = tool_execution_errors > 0
    
    # Check if model used proper tool call format
    tool_calls = extract_tool_calls_from_text(solution)
    used_tool = len(tool_calls) > 0
    
    extracted = extract_boxed_answer(solution)
    
    if extracted is None:
        # Try to find the answer in the last part of the response
        # Sometimes models don't use boxed format
        lines = solution.split('\n')
        if lines:
            last_line = lines[-1].strip()
            # Check if last line looks like an answer (short, single word/phrase)
            if len(last_line) < 50 and last_line:
                extracted = last_line
    
    # Normalize both for comparison
    extracted_norm = normalize_answer(extracted) if extracted else ""
    ground_truth_norm = normalize_answer(str(ground_truth))
    
    # Check answer correctness
    exact_match = extracted_norm == ground_truth_norm
    partial_match = (ground_truth_norm in extracted_norm or extracted_norm in ground_truth_norm) if extracted_norm else False
    
    if tool_execution_failed:
        # Tool call failed to execute 
        score = 0.0
        logger.error(f"Tool execution failed! Predicted: {extracted_norm}, Expected: {ground_truth_norm}, Errors: {tool_execution_errors}")
    elif not used_tool:
        # No tool call detected - zero reward (model must learn to use the tool)
        score = 0.0
        logger.error(f"No tool call found! Predicted: {extracted_norm}, Expected: {ground_truth_norm}")
    elif not extracted:
        # Tool used but no answer found
        score = 0.1
        logger.error(f"No boxed answer found. Solution: {solution[:200]}...")
    elif exact_match:
        # Tool used correctly and answer is correct
        score = 1.0
        logger.error(f"Correct! Predicted: {extracted_norm}, Expected: {ground_truth_norm}")
    elif partial_match:
        # Partial match
        score = 0.5
        logger.error(f"Partial match. Predicted: {extracted_norm}, Expected: {ground_truth_norm}")
    else:
        # Tool used but answer wrong - small reward for trying
        score = 0.1
        logger.error(f"Wrong answer. Predicted: {extracted_norm}, Expected: {ground_truth_norm}")
    
    return score
