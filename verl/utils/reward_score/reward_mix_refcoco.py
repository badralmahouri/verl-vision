"""
Mixed-tool reward for RefCOCO-based "mixed tools" datasets.

This reward supports two common task families:

1) Spatial tasks (IoU reward):
   - `image_bbox_tool`: reward is max IoU(pred_bbox, any expected bbox) in [0, 1]
   - `image_zoom_in_tool`: reward is IoU(pred_bbox, expected bbox) in [0, 1]
   - For bbox tasks, required non-bbox tools must appear before the bbox call.

2) Non-spatial tool tasks (binary reward):
   - reward is 1.0 iff the model:
       a) uses the expected tool(s)
       b) and, if `ground_truth` is a string, produces the correct answer in <boxed></boxed>
   - otherwise reward is 0.0

Expected tool calls should be provided via `extra_info["expected_tool_calls"]` (or
`extra_info["metadata"]["expected_tool_calls"]`) in the same format used by other
VERL tool datasets:
  [{"tool": "image_flip_tool", "parameters": {"direction": "horizontal"}}, ...]
"""

import json
import logging
import re
import sys

logger = logging.getLogger("reward_mix_refcoco")
logger.setLevel(logging.ERROR)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("[REFCOCO_MIX_REWARD] %(message)s"))
    logger.addHandler(handler)


def compute_iou(box1, box2) -> float:
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


def extract_tool_calls_from_text(text: str) -> list[dict]:
    if not text:
        return []
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls: list[dict] = []
    for m in matches:
        try:
            parsed = json.loads(m.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                tool_calls.append(parsed)
        except Exception:
            continue
    return tool_calls


def _get_tool_arguments(tool_call: dict) -> dict:
    args = tool_call.get("arguments")
    if isinstance(args, dict):
        return args
    args = tool_call.get("args")
    if isinstance(args, dict):
        return args
    return {}


def _extract_bbox_from_args(args: dict) -> list[int] | None:
    bbox = args.get("bbox_2d") or args.get("bbox")
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return [int(round(float(b))) for b in bbox]
    except Exception:
        return None


def _normalize_expected_boxes(expected) -> list[list[int]]:
    boxes: list[list[int]] = []
    if expected is None:
        return boxes
    if (
        isinstance(expected, (list, tuple))
        and len(expected) == 4
        and all(isinstance(x, (int, float)) for x in expected)
    ):
        return [[int(expected[0]), int(expected[1]), int(expected[2]), int(expected[3])]]
    if isinstance(expected, list):
        for item in expected:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                try:
                    boxes.append([int(item[0]), int(item[1]), int(item[2]), int(item[3])])
                except Exception:
                    continue
    return boxes


def extract_boxed_answer(text: str) -> str | None:
    if not text:
        return None
    patterns = [r"<boxed>(.*?)</boxed>", r"\\boxed\{(.*?)\}"]
    for pat in patterns:
        matches = re.findall(pat, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip().lower()
    return None


def _normalize_text_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[.,!?;:'\"`]+", "", text)
    text = " ".join(text.split())
    return text


def _get_expected_tool_calls(extra_info: dict | None) -> list[dict]:
    if not extra_info or not isinstance(extra_info, dict):
        return []
    calls = extra_info.get("expected_tool_calls")
    if isinstance(calls, list):
        return [c for c in calls if isinstance(c, dict)]
    md = extra_info.get("metadata")
    if isinstance(md, dict):
        calls = md.get("expected_tool_calls")
        if isinstance(calls, list):
            return [c for c in calls if isinstance(c, dict)]
    return []


def _expected_name(call: dict) -> str | None:
    name = call.get("tool") or call.get("name")
    return str(name) if name else None


def _expected_params(call: dict) -> dict:
    params = call.get("parameters") or call.get("arguments") or {}
    return params if isinstance(params, dict) else {}


def _coerce_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _values_equal(expected, actual) -> bool:
    if expected is None:
        return actual is None

    exp_num = _coerce_number(expected)
    if exp_num is not None:
        act_num = _coerce_number(actual)
        if act_num is None:
            return False
        return abs(exp_num - act_num) < 1e-6

    if isinstance(expected, str):
        return str(actual).strip().lower() == expected.strip().lower()

    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            return False
        return all(_values_equal(e, a) for e, a in zip(expected, actual))

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for k, v in expected.items():
            if k not in actual:
                return False
            if not _values_equal(v, actual[k]):
                return False
        return True

    return expected == actual


def compute_score(solution_str, ground_truth=None, data_source=None, extra_info=None, **kwargs) -> float:
    try:
        sol = str(solution_str or "")
        tool_calls = extract_tool_calls_from_text(sol)

        tool_execution_errors = 0
        if extra_info and isinstance(extra_info, dict):
            tool_execution_errors = int(extra_info.get("tool_execution_errors", 0) or 0)
        if tool_execution_errors > 0:
            logger.error(
                json.dumps({"score": 0.0, "reason": "tool_execution_failed", "tool_execution_errors": tool_execution_errors})
            )
            return 0.0

        expected_calls = _get_expected_tool_calls(extra_info)
        expected_tools = [_expected_name(c) for c in expected_calls if _expected_name(c)]
        is_bbox_task = "image_bbox_tool" in expected_tools
        is_zoom_task = (not is_bbox_task) and ("image_zoom_in_tool" in expected_tools)

        # BBox task (IoU)
        if is_bbox_task:
            bbox_index = None
            pred_bbox = None
            for idx, act in enumerate(tool_calls):
                if act.get("name") == "image_bbox_tool":
                    bbox_index = idx
                    pred_bbox = _extract_bbox_from_args(_get_tool_arguments(act))
                    break

            if pred_bbox is None or bbox_index is None:
                logger.error(json.dumps({"score": 0.0, "reason": "missing_bbox"}))
                return 0.0

            required_tools = [t for t in expected_tools if t and t != "image_bbox_tool"]
            missing_tool = False
            for tool_name in required_tools:
                found = False
                for idx, act in enumerate(tool_calls):
                    if idx >= bbox_index:
                        break
                    if act.get("name") == tool_name:
                        found = True
                        break
                if not found:
                    missing_tool = True
                    break
            if missing_tool:
                logger.error(json.dumps({"score": 0.0, "reason": "missing_required_tool_before_bbox"}))
                return 0.0

            expected_boxes = _normalize_expected_boxes(ground_truth)
            if not expected_boxes and extra_info and isinstance(extra_info, dict):
                expected_boxes = _normalize_expected_boxes(
                    extra_info.get("ground_truth_bbox") or extra_info.get("metadata", {}).get("ground_truth_bbox")
                )
            if not expected_boxes:
                # fallback to expected_tool_calls bbox if provided
                for exp in expected_calls:
                    if _expected_name(exp) == "image_bbox_tool":
                        expected_boxes = _normalize_expected_boxes(_expected_params(exp).get("bbox_2d"))
                        break

            if not expected_boxes:
                logger.error(json.dumps({"score": 0.0, "reason": "no_expected_boxes"}))
                return 0.0

            best = 0.0
            for b in expected_boxes:
                best = max(best, compute_iou(pred_bbox, b))
            logger.error(json.dumps({"score": round(best, 4), "task": "bbox"}))
            return float(best)

        # Zoom task (IoU)
        if is_zoom_task:
            expected_zoom = None
            for exp in expected_calls:
                if _expected_name(exp) == "image_zoom_in_tool":
                    expected_zoom = _extract_bbox_from_args(_expected_params(exp))
                    break
            pred_zoom = None
            for act in tool_calls:
                if act.get("name") == "image_zoom_in_tool":
                    pred_zoom = _extract_bbox_from_args(_get_tool_arguments(act))
                    break
            if pred_zoom is None or expected_zoom is None:
                logger.error(json.dumps({"score": 0.0, "reason": "missing_zoom_bbox"}))
                return 0.0
            score = compute_iou(pred_zoom, expected_zoom)
            logger.error(json.dumps({"score": round(score, 4), "task": "zoom"}))
            return float(score)

        # Tool-only task (binary)
        if isinstance(ground_truth, str):
            pred = extract_boxed_answer(sol)
            exp = extract_boxed_answer(ground_truth) or str(ground_truth)
            if _normalize_text_answer(pred) != _normalize_text_answer(exp):
                logger.error(json.dumps({"score": 0.0, "reason": "answer_mismatch", "pred": pred, "expected": exp}))
                return 0.0

        if expected_tools:
            missing = False
            for tool_name in expected_tools:
                if not any(tc.get("name") == tool_name for tc in tool_calls):
                    missing = True
                    break
            if missing:
                logger.error(json.dumps({"score": 0.0, "reason": "missing_expected_tool"}))
                return 0.0
        elif not tool_calls:
            logger.error(json.dumps({"score": 0.0, "reason": "no_tool_calls"}))
            return 0.0

        return 1.0

    except Exception as e:
        logger.error(f"compute_score error: {e}")
        return 0.0


__all__ = ["compute_score"]
