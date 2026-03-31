"""
Binary reward function for zoom/crop tool use training.
Correct <boxed>answer</boxed> = 1.0, wrong = 0.0.
No IoU, no tool-call detection, no partial credit.
"""

import json
import logging
import os
import re
import sys

logger = logging.getLogger("reward_zoom")
logger.setLevel(logging.ERROR)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.ERROR)
    h.setFormatter(logging.Formatter("[ZOOM_REWARD] %(message)s"))
    logger.addHandler(h)

_reward_log_file = None


def _get_reward_log():
    """Return a persistent file handle for per-sample reward debug logs."""
    global _reward_log_file
    if _reward_log_file is None:
        log_path = os.environ.get("REWARD_DEBUG_LOG", "/tmp/reward_zoom_debug.jsonl")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        _reward_log_file = open(log_path, "a")
    return _reward_log_file


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


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    """Binary reward: correct boxed answer = 1.0, wrong = 0.0."""
    try:
        solution = str(solution_str or "").strip()
        expected_answer = str(ground_truth or "").strip()

        # Extract boxed answer from model response
        predicted = extract_boxed_answer(solution)

        # Extract expected answer (strip boxed tags if present)
        expected_clean = re.sub(r'</?boxed>', '', expected_answer).strip().lower()
        expected_clean = re.sub(r'\\boxed\{([^}]+)\}', r'\1', expected_clean).strip().lower()

        # Binary score: case-insensitive substring match
        score = 0.0
        if predicted and expected_clean:
            pred_clean = predicted.strip().lower()
            if pred_clean == expected_clean or pred_clean in expected_clean or expected_clean in pred_clean:
                score = 1.0

        # Check if model used a tool call (for analysis only, not scoring)
        has_tool_call = "<tool_call>" in solution

        # Debug log
        response_preview = solution[:200].replace('\n', '\\n') if solution else ""
        debug_data = json.dumps({
            "score": score,
            "predicted_answer": predicted,
            "expected_answer": expected_clean,
            "has_tool_call": has_tool_call,
            "response_preview": response_preview,
        })
        try:
            log_file = _get_reward_log()
            log_file.write(debug_data + "\n")
            log_file.flush()
        except Exception:
            pass
        logger.error(debug_data)

        return score

    except Exception as e:
        logger.error(f"compute_score error: {e}")
        return 0.0


__all__ = ["compute_score"]
