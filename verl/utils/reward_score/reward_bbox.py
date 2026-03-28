"""
Reward function for bbox tool use training.
Rewards: IoU of predicted bbox vs expected + answer correctness in \\boxed{}.
Supports both Hermes (<tool_call>) and Apertus (<|tools_prefix|>) tool call formats.
Normalized to [0, 1] range.
"""

import logging
import json
import sys
import re

logger = logging.getLogger("reward_bbox")
logger.setLevel(logging.ERROR)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.ERROR)
    h.setFormatter(logging.Formatter("[BBOX_REWARD] %(message)s"))
    logger.addHandler(h)


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def extract_tool_calls_from_text(text):
    """
    Extract properly formatted tool calls from model output.
    Supports two formats:
      - Hermes: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
      - Apertus: <|tools_prefix|>[{"func_name": {args}}]<|tools_suffix|>
    Returns list of parsed tool call dicts with "name" and "arguments" keys.
    """
    if not text:
        return []

    tool_calls = []

    # Format 1: Hermes — <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    hermes_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    for match in re.findall(hermes_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                tool_calls.append(parsed)
        except (json.JSONDecodeError, ValueError):
            continue

    # Format 2: Apertus — <|tools_prefix|>[{"func_name": {args}}]<|tools_suffix|>
    apertus_pattern = r'<\|tools_prefix\|>(.*?)<\|tools_suffix\|>'
    for match in re.findall(apertus_pattern, text, re.DOTALL):
        try:
            calls = json.loads(match.strip())
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        for func_name, args in call.items():
                            tool_calls.append({
                                "name": func_name,
                                "arguments": args if isinstance(args, dict) else {},
                            })
        except (json.JSONDecodeError, ValueError):
            continue

    return tool_calls


def extract_bbox_from_tool_calls(tool_calls):
    """Extract bbox from properly parsed tool calls only."""
    for tc in tool_calls:
        args = tc.get("arguments", {})
        bbox = args.get("bbox_2d")
        if bbox and len(bbox) == 4:
            try:
                return [int(b) for b in bbox]
            except (ValueError, TypeError):
                continue
    return None


def extract_bbox_from_freetext(text):
    """
    Fallback: extract bbox coordinates from free text.
    Looks for patterns like [x1, y1, x2, y2] or bbox_2d: [x1, y1, x2, y2].
    Returns the first valid 4-int coordinate found, or None.
    """
    if not text:
        return None
    # Pattern: 4 integers in brackets, e.g. [100, 200, 300, 400]
    for match in re.finditer(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', text):
        coords = [int(match.group(i)) for i in range(1, 5)]
        # Sanity: x2 > x1 and y2 > y1, all values reasonable for an image
        if coords[2] > coords[0] and coords[3] > coords[1] and all(0 <= c <= 2000 for c in coords):
            return coords
    return None


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
    """
    Score = (IoU * 0.6 + answer_score * 0.4) normalized to [0, 1].

    Rewards proper tool calls in either Hermes or Apertus format.
    - No proper tool call -> zero reward
    - Tool execution error -> zero reward (penalize malformed calls)
    - Proper tool call with bbox -> IoU score
    - Correct boxed answer -> answer bonus
    """
    try:
        solution = str(solution_str or "").strip()
        expected_answer = str(ground_truth or "").strip()
        
        expected_bbox = None
        if extra_info:
            etc = extra_info.get("expected_tool_calls") or extra_info.get("metadata", {}).get("expected_tool_calls")
            if etc and len(etc) > 0:
                expected_bbox = etc[0].get("parameters", {}).get("bbox_2d")
        
        tool_calls = extract_tool_calls_from_text(solution)
        used_real_tool = len(tool_calls) > 0
        
        tool_execution_errors = 0
        if extra_info:
            tool_execution_errors = extra_info.get("tool_execution_errors", 0)
        
        tool_execution_failed = tool_execution_errors > 0
        
        # Extract bbox from proper tool calls first, then fallback to free text
        pred_bbox = extract_bbox_from_tool_calls(tool_calls) if used_real_tool else None
        used_freetext = False
        if not pred_bbox:
            pred_bbox = extract_bbox_from_freetext(solution)
            if pred_bbox:
                used_freetext = True

        # Extract boxed answer
        boxed_answer = extract_boxed_answer(solution)

        # Compute component scores
        iou_score = 0.0
        answer_score = 0.0

        if pred_bbox and expected_bbox:
            iou_score = compute_iou(pred_bbox, expected_bbox)

        # Answer check
        if boxed_answer and expected_answer:
            exp_clean = re.sub(r'</?boxed>', '', expected_answer).strip().lower()
            pred_clean = boxed_answer.strip().lower()
            if pred_clean == exp_clean or pred_clean in exp_clean or exp_clean in pred_clean:
                answer_score = 1.0

        # Final normalized score [0, 1]
        # Tool call with bbox: full credit (0.6 * IoU + 0.4 * answer)
        # Free-text bbox: partial credit (0.4 * IoU + 0.3 * answer) — incentivizes tool format
        # Tool call without bbox: small credit
        # Nothing: zero
        if tool_execution_failed: score = 0.0
        elif used_real_tool and pred_bbox: score = (iou_score * 0.6) + (answer_score * 0.4)
        elif used_real_tool: score = 0.1 + (answer_score * 0.3)
        elif used_freetext and pred_bbox: score = (iou_score * 0.4) + (answer_score * 0.3)
        elif boxed_answer: score = answer_score * 0.05  # Tiny signal for correct answer without bbox
        else: score = 0.0

        # Log first 200 chars of response for debugging model output
        response_preview = solution[:200].replace('\n', '\\n') if solution else ""
        logger.error(json.dumps({
            "score": round(score, 4),
            "iou": round(iou_score, 4),
            "answer_score": round(answer_score, 4),
            "num_tool_calls": len(tool_calls),
            "freetext_bbox": used_freetext,
            "pred_bbox": pred_bbox,
            "expected_bbox": expected_bbox,
            "response_preview": response_preview,
        }))
        
        return score
        
    except Exception as e:
        logger.error(f"compute_score error: {e}")
        return 0.0


__all__ = ["compute_score"]
