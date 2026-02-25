"""
Reward function for crop/zoom tool evaluation.
Measures IoU between predicted crop bbox and expected bbox.
"""

import logging
import json
import sys
import re

logger = logging.getLogger("reward_crop")
logger.setLevel(logging.ERROR)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.ERROR)
    h.setFormatter(logging.Formatter("[CROP_REWARD] %(message)s"))
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
    """Extract tool calls from response text."""
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


def extract_bbox_from_tool_calls(tool_calls, tool_name="image_zoom_in_tool"):
    """Extract bbox from tool calls for specific tool."""
    for tc in tool_calls:
        if tc.get("name") == tool_name:
            args = tc.get("arguments", {})
            bbox = args.get("bbox_2d")
            if bbox and len(bbox) == 4:
                try:
                    return [int(b) for b in bbox]
                except (ValueError, TypeError):
                    continue
    return None


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    """Score a crop/zoom task output.

    Returns a float in [0, 1]. solution_str is the model output; ground_truth
    is the expected bbox; `extra_info` may contain tool execution metadata.
    """
    try:
        solution = str(solution_str or "").strip()
        
        # Get expected bbox from extra_info
        expected_bbox = None
        if extra_info:
            etc = extra_info.get("expected_tool_calls") or extra_info.get("metadata", {}).get("expected_tool_calls")
            if etc and len(etc) > 0:
                expected_bbox = etc[0].get("parameters", {}).get("bbox_2d")
        
        if expected_bbox is None:
            logger.error("No expected bbox found in extra_info")
            return 0.0
        
        # Check for tool execution errors
        tool_execution_errors = 0
        if extra_info:
            tool_execution_errors = extra_info.get("tool_execution_errors", 0)
        
        if tool_execution_errors > 0:
            logger.error(json.dumps({
                "score": 0.0,
                "iou": 0.0,
                "tool_execution_failed": True,
                "tool_execution_errors": tool_execution_errors,
                "expected_bbox": expected_bbox
            }))
            return 0.0
        
        tool_calls = extract_tool_calls_from_text(solution)
        used_zoom_tool = any(tc.get("name") == "image_zoom_in_tool" for tc in tool_calls)
        
        # Extract predicted bbox
        pred_bbox = extract_bbox_from_tool_calls(tool_calls, "image_zoom_in_tool")
        
        # Compute IoU
        iou_score = 0.0
        if pred_bbox and expected_bbox:
            iou_score = compute_iou(pred_bbox, expected_bbox)
        
        if used_zoom_tool and pred_bbox:
            score = iou_score
        elif used_zoom_tool:
            # Used tool but invalid bbox
            score = 0.1
        else:
            # Didn't use tool
            score = 0.0
        
        logger.error(json.dumps({
            "score": round(score, 4),
            "iou": round(iou_score, 4),
            "used_zoom_tool": used_zoom_tool,
            "num_tool_calls": len(tool_calls),
            "pred_bbox": pred_bbox,
            "expected_bbox": list(expected_bbox) if expected_bbox else None,
            "solution_preview": solution[:100] if solution else None
        }))
        
        return score
        
    except Exception as e:
        logger.error(f"compute_score error: {e}")
        return 0.0


__all__ = ["compute_score"]
