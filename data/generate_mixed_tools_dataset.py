#!/usr/bin/env python3
"""
Generate a RefCOCO dataset mixing multiple image tools (rotate, flip, zoom, bbox, blur/unblur).

The generator creates two task families:

1) BBox localization tasks (IoU reward):
   - The prompt provides a referring expression.
   - The input image is corrupted (rotate/flip/blur).
   - The model must use the correct tool to restore the image before
     calling `image_bbox_tool` to localize the object.
   - Reward is computed as IoU against the RefCOCO ground-truth bbox list.

2) Tool-only tasks (binary reward):
   - The prompt asks for exactly one tool call (rotate/flip/blur/unblur) with
     explicit parameters and then a final `<boxed>done</boxed>` answer.
   - Reward is 1/0 based on correct tool usage + answer.

This dataset is intended to be scored with `verl.utils.reward_score.reward_mix_refcoco`
using `data_source="refcoco_mixed_tools"`.

The generator saves transformed inputs under:
  <output_dir>/mixed_tools/images
If --copy_images is set, clean originals are copied to:
  <output_dir>/mixed_tools/images_clean

Usage:
  python data/generate_mixed_tools_dataset.py --data_dir <refcoco_dir> --images_dir <coco_images_dir> --output_dir <out_dir>
"""

import os
import sys
import random
import argparse
import importlib.util
from collections import Counter
from uuid import uuid4

from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageFilter


def load_refcoco_helpers(path):
    spec = importlib.util.spec_from_file_location("generate_real_world", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ALL_TOOLS = [
    "image_rotate_tool",
    "image_flip_tool",
    "image_zoom_in_tool",
    "image_bbox_tool",
    "image_blur_tool",
    "image_unblur_tool",
]


SYSTEM_PROMPT = """You are a helpful assistant that uses tools to analyze images.

Tool calls MUST be formatted exactly as:
<tool_call>
{"name": "<tool_name>", "arguments": {...}}
</tool_call>

If a final text answer is requested, put it in <boxed></boxed>.
"""


def _load_rgb_image(path):
    try:
        img = Image.open(path)
    except Exception:
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _save_image(img, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    try:
        img.save(out_path, quality=95)
    except Exception:
        img.save(out_path)
    return os.path.abspath(out_path)


def _apply_blur(img, blur_type, intensity):
    radius = max(1, int(intensity)) * 2
    if blur_type == "gaussian":
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    if blur_type == "box":
        return img.filter(ImageFilter.BoxBlur(radius=radius))
    return img


def create_mixed_sample(ref_data, images_dir, images_out_dir, clean_images_out_dir=None, min_steps=2, max_steps=3):
    image_path = os.path.join(images_dir, ref_data["image_file"])
    if not os.path.exists(image_path):
        return None
    if not ref_data.get("sentences"):
        return None

    orig_img = _load_rgb_image(image_path)
    if orig_img is None:
        return None

    gt_bbox = ref_data.get("bbox", [])
    gt_bboxes = ref_data.get("all_valid_bboxes") or ([gt_bbox] if gt_bbox else [])
    if not gt_bboxes:
        return None

    referring_expr = random.choice(ref_data["sentences"])

    # Decide between bbox localization (IoU) and tool-only (binary) tasks.
    task_kind = random.choices(["bbox", "tool_only"], weights=[0.75, 0.25])[0]

    if task_kind == "bbox":
        expected_calls = []
        user_steps = []
        corruption_type = random.choice(["rotate", "flip", "blur"])
        corruption_params = {}
        restoration_params = {}
        if corruption_type == "rotate":
            rotate_angle = random.choice([90, 180, 270])
            corrupted_img = orig_img.rotate(rotate_angle, expand=True)
            restore_angle = (360 - rotate_angle) % 360
            if restore_angle == 0:
                restore_angle = 360
            expected_calls.append({"tool": "image_rotate_tool", "parameters": {"angle": int(restore_angle)}})
            user_steps.append("- The image is rotated. Use `image_rotate_tool` to restore it.")
            corruption_params = {"angle": int(rotate_angle)}
            restoration_params = {"angle": int(restore_angle)}
        elif corruption_type == "flip":
            direction = random.choice(["horizontal", "vertical"])
            if direction == "horizontal":
                corrupted_img = orig_img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                corrupted_img = orig_img.transpose(Image.FLIP_TOP_BOTTOM)
            expected_calls.append({"tool": "image_flip_tool", "parameters": {"direction": direction}})
            user_steps.append("- The image is flipped. Use `image_flip_tool` to restore it.")
            corruption_params = {"direction": direction}
            restoration_params = {"direction": direction}
        else:
            blur_type = random.choice(["gaussian", "box"])
            intensity = random.randint(2, 6)
            corrupted_img = _apply_blur(orig_img, blur_type, intensity)
            unblur_type = random.choice(["sharpen", "unsharp_mask", "edge_enhance"])
            unblur_intensity = random.randint(1, 3)
            expected_calls.append(
                {"tool": "image_unblur_tool", "parameters": {"blur_type": unblur_type, "intensity": int(unblur_intensity)}}
            )
            user_steps.append("- The image is blurred. Use `image_unblur_tool` to sharpen it.")
            corruption_params = {"blur_type": blur_type, "intensity": int(intensity)}
            restoration_params = {"blur_type": unblur_type, "intensity": int(unblur_intensity)}

        expected_calls.append({"tool": "image_bbox_tool", "parameters": {"bbox_2d": gt_bboxes[0]}})
        user_steps.append(f"- Then draw a bounding box around: {referring_expr!r} using `image_bbox_tool`.")

        token = uuid4().hex[:8]
        corrupt_name = f"ref_{ref_data['ref_id']}_bbox_{corruption_type}_{token}.jpg"
        corrupt_path = _save_image(corrupted_img, images_out_dir, corrupt_name)
        if clean_images_out_dir:
            clean_name = f"ref_{ref_data['ref_id']}_clean_{token}.jpg"
            clean_path = _save_image(orig_img, clean_images_out_dir, clean_name)
        else:
            clean_path = os.path.abspath(image_path)

        # Make all tools available, but ensure bbox uses the clean image.
        tools_kwargs = {t: {"create_kwargs": {"image": corrupt_path}} for t in ALL_TOOLS}
        tools_kwargs["image_bbox_tool"] = {"create_kwargs": {"image": clean_path}}
        available_tools = list(ALL_TOOLS)

        user_content = (
            "<image>\n"
            "Follow these steps using tool calls:\n"
            + "\n".join(user_steps)
            + "\n\n"
            "Return the bbox as a tool call."
        )
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        answer = ""
        reward_model = {
            "style": "rule",
            "type": "refcoco_iou",
            "ground_truth_bbox": gt_bboxes,
            # Keep this as a string for pyarrow schema consistency across mixed tasks.
            "ground_truth": "",
        }
    else:
        expected_calls = []
        user_steps = []
        token = uuid4().hex[:8]
        tool_image_name = f"ref_{ref_data['ref_id']}_tool_{token}.jpg"
        tool_image_path = _save_image(orig_img, images_out_dir, tool_image_name)
        tools_kwargs = {t: {"create_kwargs": {"image": tool_image_path}} for t in ALL_TOOLS}
        available_tools = list(ALL_TOOLS)

        tool_name = random.choice(["image_rotate_tool", "image_flip_tool", "image_blur_tool", "image_unblur_tool"])
        params = {}
        if tool_name == "image_rotate_tool":
            angle = random.choice([90, 180, 270])
            params = {"angle": int(angle)}
            user_steps.append(f"- Rotate the image by {angle} degrees using `image_rotate_tool`.")
        elif tool_name == "image_flip_tool":
            direction = random.choice(["horizontal", "vertical"])
            params = {"direction": direction}
            user_steps.append(f"- Flip the image {direction}ly using `image_flip_tool`.")
        elif tool_name == "image_blur_tool":
            blur_type = random.choice(["gaussian", "box"])
            intensity = random.randint(1, 5)
            params = {"blur_type": blur_type, "intensity": int(intensity)}
            user_steps.append(f"- Apply blur `blur_type`={blur_type!r}, `intensity`={intensity} using `image_blur_tool`.")
        else:
            blur_type = random.choice(["sharpen", "unsharp_mask", "edge_enhance"])
            intensity = random.randint(1, 3)
            params = {"blur_type": blur_type, "intensity": int(intensity)}
            user_steps.append(
                f"- Apply unblur `blur_type`={blur_type!r}, `intensity`={intensity} using `image_unblur_tool`."
            )

        expected_calls.append({"tool": tool_name, "parameters": params})
        user_content = (
            "<image>\n"
            "Follow these steps using tool calls (no extra tool calls):\n"
            + "\n".join(user_steps)
            + "\n\n"
            "Then reply with <boxed>done</boxed>."
        )
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        answer = "<boxed>done</boxed>"
        reward_model = {"style": "rule", "ground_truth": answer}
        corruption_type = "none"
        corruption_params = {}
        restoration_params = {}
        corrupt_path = tool_image_path
        clean_path = tool_image_path

    metadata = {
        "ref_id": ref_data["ref_id"],
        "image_id": ref_data["image_id"],
        "category": ref_data.get("category", "object"),
        "task": "mixed_tools",
        "task_kind": task_kind,
        "referring_expression": referring_expr,
        "ground_truth_bbox": gt_bboxes,
        "corruption_type": corruption_type,
        "corruption_params": corruption_params,
        "restoration_params": restoration_params,
        "corrupt_image": corrupt_path,
        "clean_image": clean_path,
    }

    extra_info = {
        "expected_tool_calls": expected_calls,
        "metadata": metadata,
        "tools_kwargs": tools_kwargs,
    }

    return {
        "prompt": prompt,
        "answer": answer,
        "images": [{"image": corrupt_path}],
        "metadata": metadata,
        "data_source": "refcoco_mixed_tools",
        "reward_model": reward_model,
        "tools_kwargs": tools_kwargs,
        "extra_info": extra_info,
        "agent_name": "tool_agent",
        "available_tools": available_tools,
    }


def save_as_parquet(rows, output_dir, split_name="train"):
    os.makedirs(output_dir, exist_ok=True)

    pa_rows = []
    for r in rows:
        pa_rows.append(
            {
                "prompt": r["prompt"],
                "answer": r["answer"],
                "images": r["images"],
                "metadata": r.get("metadata", {}),
                "extra_info": r.get("extra_info", {}),
                "data_source": r.get("data_source", "refcoco_mixed_tools"),
                "reward_model": (r.get("reward_model") or None),
                "tools_kwargs": r.get("tools_kwargs", {}),
                "available_tools": r.get("available_tools", []),
                "agent_name": r.get("agent_name", "tool_agent"),
            }
        )

    table = pa.Table.from_pylist(pa_rows)
    out_path = os.path.join(output_dir, f"{split_name}.parquet")
    pq.write_table(table, out_path)
    return out_path, len(pa_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate RefCOCO mixed-tools dataset for VERL")
    parser.add_argument("--data_dir", type=str, default="data/refcoco", help="Directory where RefCOCO extracted files live")
    parser.add_argument("--images_dir", type=str, default="data/coco_images/train2014", help="Directory containing COCO train images")
    parser.add_argument("--output_dir", type=str, default="data/refcoco_mixed", help="Output directory for parquet files")
    parser.add_argument("--max_samples", type=int, default=4000, help="Maximum samples for training")
    parser.add_argument("--test_samples", type=int, default=500, help="Maximum samples for testing")
    parser.add_argument("--copy_images", action="store_true", default=False, help="Copy clean images into output (used by bbox tool)")
    parser.add_argument("--min_pipeline_steps", type=int, default=2, help="Minimum number of tool steps per sample")
    parser.add_argument("--max_pipeline_steps", type=int, default=3, help="Maximum number of tool steps per sample")
    parser.add_argument("--refs_pickle", type=str, default=None, help="Optional path to refs pickled file (overrides) ")
    args = parser.parse_args()

    gen_path = os.path.join(os.path.dirname(__file__), "generate_real_world.py")
    if not os.path.exists(gen_path):
        gen_path = os.path.join(os.getcwd(), "data", "generate_real_world.py")
    helpers = load_refcoco_helpers(gen_path)

    data_dir = args.data_dir
    images_dir = args.images_dir
    if not os.path.exists(images_dir):
        print(f"COCO images not found at {images_dir}. Please run the COCO download or point --images_dir correctly.")
        sys.exit(1)

    split_by = "unc"
    refs_pickle = args.refs_pickle or os.path.join(data_dir, f"refs({split_by}).p")
    instances_file = os.path.join(data_dir, "instances.json")
    if not os.path.exists(refs_pickle) or not os.path.exists(instances_file):
        print("RefCOCO refs or instances.json not found. Please run the RefCOCO generator first.")
        sys.exit(1)

    loader = helpers.RefCOCOLoader(data_dir, dataset="refcoco", splitBy=split_by)

    train_refs = loader.get_split("train")
    random.shuffle(train_refs)
    if args.max_samples:
        train_refs = train_refs[: args.max_samples]

    val_refs = loader.get_split("val")
    random.shuffle(val_refs)
    if args.test_samples:
        val_refs = val_refs[: args.test_samples]

    out_dir = os.path.join(args.output_dir, "mixed_tools")
    images_out_dir = os.path.join(out_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)
    clean_images_out_dir = None
    if args.copy_images:
        clean_images_out_dir = os.path.join(out_dir, "images_clean")
        os.makedirs(clean_images_out_dir, exist_ok=True)

    print(f"Generating {len(train_refs)} train and {len(val_refs)} val mixed-tool samples...")
    train_rows = []
    test_rows = []
    tool_usage = Counter()

    for ref in tqdm(train_refs, desc="Processing train"):
        ref_data = loader.get_ref_data(ref)
        if ref_data is None:
            continue
        sample = create_mixed_sample(
            ref_data,
            images_dir,
            images_out_dir,
            clean_images_out_dir,
            args.min_pipeline_steps,
            args.max_pipeline_steps,
        )
        if not sample:
            continue
        train_rows.append(sample)
        for c in sample.get("extra_info", {}).get("expected_tool_calls", []):
            tool_usage[c.get("tool")] += 1

    for ref in tqdm(val_refs, desc="Processing val"):
        ref_data = loader.get_ref_data(ref)
        if ref_data is None:
            continue
        sample = create_mixed_sample(
            ref_data,
            images_dir,
            images_out_dir,
            clean_images_out_dir,
            args.min_pipeline_steps,
            args.max_pipeline_steps,
        )
        if not sample:
            continue
        test_rows.append(sample)
        for c in sample.get("extra_info", {}).get("expected_tool_calls", []):
            tool_usage[c.get("tool")] += 1

    train_path, train_count = save_as_parquet(train_rows, out_dir, "train")
    test_path, test_count = save_as_parquet(test_rows, out_dir, "test")

    print(f"Saved Train ({train_count}): {train_path}")
    print(f"Saved Test  ({test_count}): {test_path}")
    print("Tool usage summary:")
    for t, c in tool_usage.most_common():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
