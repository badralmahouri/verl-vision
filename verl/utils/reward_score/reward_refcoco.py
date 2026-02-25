"""
Reward for RefCOCO bbox/crop tasks.
- Accepts properly formatted <tool_call>...</tool_call> blocks and extracts bbox (arguments.bbox_2d or bbox)
- If multiple expected bboxes are provided, compute IoU against each and take the max
- Returns IoU in [0,1] when a valid tool call was used; 0 otherwise
- Penalizes tool execution errors (if extra_info reports errors)

Compatible with the existing Verl reward manager conventions.
"""

import logging
import json
import sys
import re

logger = logging.getLogger("reward_refcoco")
logger.setLevel(logging.ERROR)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.ERROR)
    h.setFormatter(logging.Formatter("[REFCOCO_REWARD] %(message)s"))
    logger.addHandler(h)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return float(inter) / union if union > 0 else 0.0


def extract_tool_calls_from_text(text):
    if not text:
        return []
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for m in matches:
        try:
            parsed = json.loads(m.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                tool_calls.append(parsed)
        except Exception:
            continue
    return tool_calls


def extract_bbox_from_tool_calls(tool_calls):
    for tc in tool_calls:
        args = tc.get("arguments", {})
        bbox = args.get("bbox_2d") or args.get("bbox")
        if bbox and len(bbox) == 4:
            try:
                return [int(b) for b in bbox]
            except Exception:
                continue
    return None


def _normalize_expected_boxes(expected):
    boxes = []
    if expected is None:
        return boxes
    if isinstance(expected, (list, tuple)) and len(expected) == 4 and all(isinstance(x, (int, float)) for x in expected):
        boxes.append([int(expected[0]), int(expected[1]), int(expected[2]), int(expected[3])])
        return boxes
    if isinstance(expected, list):
        for item in expected:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                try:
                    boxes.append([int(item[0]), int(item[1]), int(item[2]), int(item[3])])
                except Exception:
                    continue
    return boxes


def compute_score(solution_str, ground_truth=None, data_source=None, extra_info=None, **kwargs):
    """
    solution_str: model output text (may contain <tool_call> blocks)
    ground_truth / extra_info: expected bboxes list under keys:
      - extra_info['metadata']['ground_truth_bbox']
      - extra_info['ground_truth_bbox']
      - reward_model['ground_truth_bbox'] (handled by reward manager)

    Returns IoU max over expected bboxes if a proper tool call provided, 0 otherwise.
    """
    try:
        sol = str(solution_str or "")

        expected_list = None
        if extra_info and isinstance(extra_info, dict):
            expected_list = extra_info.get("ground_truth_bbox") or extra_info.get("metadata", {}).get("ground_truth_bbox")
            # some generators may put reward model under extra_info
            if not expected_list:
                rm = extra_info.get("reward_model")
                if isinstance(rm, dict):
                    expected_list = rm.get("ground_truth_bbox")
        if expected_list is None and ground_truth is not None:
            expected_list = ground_truth

        boxes = _normalize_expected_boxes(expected_list)
        tool_calls = extract_tool_calls_from_text(sol)
        used_tool = len(tool_calls) > 0

        tool_execution_errors = 0
        if extra_info and isinstance(extra_info, dict):
            tool_execution_errors = extra_info.get("tool_execution_errors", 0)

        if tool_execution_errors > 0:
            logger.error(json.dumps({"msg": "tool execution failed", "errors": tool_execution_errors}))
            return 0.0

        pred_bbox = extract_bbox_from_tool_calls(tool_calls) if used_tool else None

        if not used_tool or pred_bbox is None:
            logger.error(json.dumps({"score": 0.0, "reason": "no_valid_tool_call"}))
            return 0.0

        if not boxes:
            logger.error(json.dumps({"score": 0.0, "reason": "no_expected_boxes"}))
            return 0.0

        best = 0.0
        for b in boxes:
            iou = compute_iou(pred_bbox, b)
            if iou > best:
                best = iou

        logger.error(json.dumps({"score": round(best, 4), "pred_bbox": pred_bbox, "best_iou": round(best, 4)}))
        return float(best)

    except Exception as e:
        logger.error(f"compute_score error: {e}")
        return 0.0


__all__ = ["compute_score"]
