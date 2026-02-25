"""
Reward function for line tool use training.
Rewards: Endpoint accuracy of predicted line vs expected + answer correctness in \\boxed{}.
ONLY rewards ACTUAL tool use via <tool_call>...</tool_call> format.
Normalized to [0, 1] range.
"""

import logging
import json
import sys
import re
import math

logger = logging.getLogger("reward_line")
logger.setLevel(logging.ERROR)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.ERROR)
    h.setFormatter(logging.Formatter("[LINE_REWARD] %(message)s"))
    logger.addHandler(h)


def compute_endpoint_similarity(pred_start, pred_end, expected_start, expected_end, tolerance=50.0):
    """
    Compute similarity between predicted and expected line endpoints.
    Returns a score in [0, 1] based on normalized distance.
    
    Handles line symmetry: A→B is equivalent to B→A.
    
    Args:
        tolerance: Pixel tolerance for normalization (default 50.0)
    """
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Check both directions: A→B and B→A (lines are symmetric)
    # Direction 1: pred_start→expected_start, pred_end→expected_end
    start_dist_1 = distance(pred_start, expected_start)
    end_dist_1 = distance(pred_end, expected_end)
    avg_dist_1 = (start_dist_1 + end_dist_1) / 2.0
    
    # Direction 2: pred_start→expected_end, pred_end→expected_start (reversed)
    start_dist_2 = distance(pred_start, expected_end)
    end_dist_2 = distance(pred_end, expected_start)
    avg_dist_2 = (start_dist_2 + end_dist_2) / 2.0
    
    # Use the better match (smaller distance)
    avg_dist = min(avg_dist_1, avg_dist_2)
    
    # Normalize using exponential decay with configurable tolerance
    # Perfect match = 0 distance -> score 1.0
    # Large distance (>tolerance) -> score approaches 0
    similarity = math.exp(-avg_dist / tolerance)
    
    return similarity


def extract_tool_calls_from_text(text):
    """
    Extract ONLY properly formatted tool calls using <tool_call>...</tool_call> tags.
    Returns list of parsed tool call dicts, or empty list if none found.
    """
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


def extract_line_from_tool_calls(tool_calls):
    """Extract line endpoints from properly parsed tool calls only."""
    for tc in tool_calls:
        args = tc.get("arguments", {})
        start = args.get("start")
        end = args.get("end")
        if start and end and len(start) == 2 and len(end) == 2:
            try:
                return [float(start[0]), float(start[1])], [float(end[0]), float(end[1])]
            except (ValueError, TypeError):
                continue
    return None, None


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} or <boxed>...</boxed>."""
    if not text:
        return None
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'<boxed>(.*?)</boxed>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def normalize_answer(ans):
    """Normalize answer for comparison - handles numbers and text."""
    if not ans:
        return ""
    ans_clean = str(ans).strip().lower()
    # Remove common wrappers
    ans_clean = re.sub(r'</?boxed>', '', ans_clean)
    ans_clean = re.sub(r'\\boxed\{|\}', '', ans_clean)
    return ans_clean.strip()


def answers_match(predicted, expected):
    """
    Check if answers match - handles both numeric and text answers.
    For v1: numeric tolerance (±10%)
    For v2: exact text match (e.g., "success")
    """
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    # Exact match
    if pred_norm == exp_norm:
        return True
    
    # Substring match
    if pred_norm in exp_norm or exp_norm in pred_norm:
        return True
    
    # Try numeric comparison with tolerance
    try:
        pred_num = float(re.sub(r'[^\d.-]', '', pred_norm))
        exp_num = float(re.sub(r'[^\d.-]', '', exp_norm))
        tolerance = abs(exp_num * 0.1)  # 10% tolerance
        if abs(pred_num - exp_num) <= max(tolerance, 5):  # At least 5 pixels
            return True
    except (ValueError, TypeError):
        pass
    
    return False


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    """Score a line task output.

    Returns a float in [0, 1]. solution_str is the model output; ground_truth
    is the expected answer (or coords). extra_info may contain expected tool
    calls, metadata, and tool execution info.
    """
    try:
        solution = str(solution_str or "").strip()
        expected_answer = str(ground_truth or "").strip()
        
        # Debug: Log what we received
        logger.error(f"[DEBUG] extra_info type: {type(extra_info)}, keys: {list(extra_info.keys()) if extra_info and isinstance(extra_info, dict) else 'N/A'}")
        
        # Get expected line from extra_info - check multiple possible locations
        expected_start = None
        expected_end = None
        if extra_info and isinstance(extra_info, dict):
            # Primary location: directly in extra_info
            etc = extra_info.get("expected_tool_calls")
            # Fallback: nested in metadata
            if not etc:
                etc = extra_info.get("metadata", {}).get("expected_tool_calls")
            # Fallback: in reward_model
            if not etc:
                etc = extra_info.get("reward_model", {}).get("expected_tool_calls")
            
            if etc and len(etc) > 0:
                params = etc[0].get("parameters", {})
                expected_start = params.get("start")
                expected_end = params.get("end")
        
        logger.error(f"[DEBUG] Expected coords: start={expected_start}, end={expected_end}")
        
        # Extract ONLY properly formatted tool calls
        tool_calls = extract_tool_calls_from_text(solution)
        used_real_tool = len(tool_calls) > 0
        
        # Check for tool execution errors
        tool_execution_errors = 0
        if extra_info:
            tool_execution_errors = extra_info.get("tool_execution_errors", 0)
        
        tool_execution_failed = tool_execution_errors > 0
        
        # Extract line from proper tool calls only
        pred_start, pred_end = extract_line_from_tool_calls(tool_calls) if used_real_tool else (None, None)
        
        # Extract boxed answer
        boxed_answer = extract_boxed_answer(solution)
        
        # Get coordinate tolerance from extra_info (v2 dataset feature)
        coord_tolerance = 50.0  # default
        if extra_info:
            metadata = extra_info.get("metadata", {})
            coord_tolerance = metadata.get("coordinate_tolerance", 50.0)
        
        # Compute component scores
        endpoint_score = 0.0
        answer_score = 0.0
        
        if pred_start and pred_end and expected_start and expected_end:
            # Use dynamic tolerance from metadata
            endpoint_score = compute_endpoint_similarity(
                pred_start, pred_end, expected_start, expected_end, tolerance=coord_tolerance
            )
        
        # Answer check - use improved matching
        if boxed_answer and expected_answer:
            if answers_match(boxed_answer, expected_answer):
                answer_score = 1.0
        
        # Final normalized score [0, 1]
        # CRITICAL: Zero reward for tool execution errors or no tool use
        if tool_execution_failed:
            # Tool call failed to execute - zero reward to train model to avoid malformed calls
            score = 0.0
        elif used_real_tool and pred_start and pred_end:
            # Full reward: tool use with line + answer
            # For v2 (answer="success"), endpoint matters more
            is_v2_style = normalize_answer(expected_answer) == "success"
            if is_v2_style:
                # v2: 80% endpoint accuracy, 20% answer confirmation
                score = (endpoint_score * 0.8) + (answer_score * 0.2)
            else:
                # v1: 60% endpoint accuracy, 40% answer (distance/measurement)
                score = (endpoint_score * 0.6) + (answer_score * 0.4)
        elif used_real_tool:
            # Partial reward: used tool format but no valid line
            score = 0.2 + (answer_score * 0.3)
        else:
            # NO reward without tool use - this is critical!
            score = 0.0
        
        logger.error(json.dumps({
            "score": round(score, 4),
            "endpoint_score": round(endpoint_score, 4),
            "answer_score": round(answer_score, 4),
            "coord_tolerance": coord_tolerance,
            "is_v2_style": normalize_answer(expected_answer) == "success",
            "used_real_tool": used_real_tool,
            "tool_execution_failed": tool_execution_failed,
            "tool_execution_errors": tool_execution_errors,
            "num_tool_calls": len(tool_calls),
            "pred_start": pred_start,
            "pred_end": pred_end,
            "expected_start": expected_start,
            "expected_end": expected_end,
            "boxed_answer": boxed_answer,
            "expected_answer": expected_answer[:50] if expected_answer else None,
            "solution_preview": solution[:100] if solution else None
        }))
        
        return score
        
    except Exception as e:
        logger.error(f"compute_score error: {e}")
        return 0.0


__all__ = ["compute_score"]
