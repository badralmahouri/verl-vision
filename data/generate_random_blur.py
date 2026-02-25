#!/usr/bin/env python3
"""
Generate blur reveal tasks that require the unblur (sharpen) tool.

Note: these tasks must call the image_unblur_tool (unblur/sharpen variants).
The key insight is that the model MUST use the unblur/sharpen tool to answer
correctly — the visible content is only revealed after unblurring.

This differs from the plain blur tool which applies blurring ops; these
examples explicitly expect the unblur variant (see verl/tools/image_blur_tool.py).
"""

import os
import random
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from collections import Counter
import json



FONT_PATH = "data/arial.ttf"  


IMG_WIDTH = 800
IMG_HEIGHT = 600

BACKGROUND_COLORS = [
    (255, 255, 255),
    (240, 240, 240),
    (255, 250, 240),
]

TEXT_COLORS = [
    (0, 0, 0),
    (220, 20, 60),
    (0, 100, 0),
    (0, 0, 139),
]

# Secret codes 
SECRET_CODES = [
    "ALPHA1", "BETA2", "GAMMA3", "DELTA4", "EPSILON5",
    "CODE42", "KEY99", "PASS77", "SECRET0", "HIDDEN8",
    "UNLOCK", "ACCESS", "VERIFY", "CONFIRM", "REVEAL",
    "PUZZLE", "ANSWER", "SOLVED", "WINNER", "MASTER",
]

# Shapes with specific positions
SHAPE_NAMES = ["circle", "square", "triangle", "star"]
POSITIONS = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]

SHAPE_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
}

WORDS_FILE = os.path.join("data", "words.txt")
if not os.path.exists(WORDS_FILE):
    raise FileNotFoundError(f"Required word list not found: {WORDS_FILE}")
with open(WORDS_FILE, "r", encoding="utf-8") as f:
    WORD_LIST = [w.strip().upper() for w in f if w.strip()]
if not WORD_LIST:
    raise ValueError(f"Word list is empty: {WORDS_FILE}")

SYSTEM_PROMPT = """You are a helpful assistant that analyzes images using tools.

The image you see is heavily blurred. To answer the question, you MUST:
1. Use the image_unblur_tool (unblur/sharpen variants) with `blur_type="sharpen"` to reveal the hidden content
2. After seeing the sharpened result, provide your answer directly (just the answer, no tags)

Example tool call:
<tool_call>
{"name": "image_unblur_tool", "arguments": {"blur_type": "sharpen", "intensity": 8}}
</tool_call>

You CANNOT answer correctly without unblurring/sharpening the image first."""


def get_font(size):
    """Load required primary font or raise.

    The repository requires `FONT_PATH` to be present; fail fast if missing.
    """
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception as e:
        raise RuntimeError(f"Failed to load required font at {FONT_PATH}: {e}")


def apply_extreme_blur(image, intensity=20):
    """Apply very strong blur - content should be COMPLETELY unrecognizable."""
    # Apply multiple blur passes
    result = image
    for _ in range(3):
        result = result.filter(ImageFilter.GaussianBlur(radius=intensity))
    return result


def generate_secret_code_scene():
    """
    Generate a scene with a SECRET CODE that is only visible after sharpening.
    The question asks specifically about the hidden code.
    """
    bg_color = random.choice(BACKGROUND_COLORS)
    text_color = random.choice(TEXT_COLORS)
    secret_code = random.choice(SECRET_CODES)
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    font_size = 80
    font = get_font(font_size)
    
    # Draw the secret code in the center
    bbox = draw.textbbox((0, 0), secret_code, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (IMG_WIDTH - text_width) // 2
    y = (IMG_HEIGHT - text_height) // 2
    
    draw.text((x, y), secret_code, fill=text_color, font=font)
    
    label_font = get_font(40)
    draw.text((x, y - 60), "SECRET CODE:", fill=(128, 128, 128), font=label_font)
    
    blur_intensity = random.randint(20, 30)
    blurred_img = apply_extreme_blur(img, blur_intensity)
    
    question = "This image contains a hidden secret code that has been blurred. Use the image_unblur_tool with sharpen to reveal and read the secret code. What is the secret code?"
    answer = str(secret_code)

    expected_tool_calls = [{"name": "image_unblur_tool", "blur_type": "sharpen"}]
    
    metadata = {
        "task_type": "secret_code_reveal",
        "blur_intensity": blur_intensity,
        "secret_code": secret_code
    }
    
    return blurred_img, question, answer, expected_tool_calls, metadata


def generate_hidden_shape_scene():
    """
    Generate a scene with a hidden shape. Question asks about shape AFTER sharpening.
    """
    bg_color = random.choice(BACKGROUND_COLORS)
    shape_name = random.choice(SHAPE_NAMES)
    color_name, color_rgb = random.choice(list(SHAPE_COLORS.items()))
    position = random.choice(POSITIONS)
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate position
    positions_coords = {
        "top-left": (IMG_WIDTH // 4, IMG_HEIGHT // 4),
        "top-right": (3 * IMG_WIDTH // 4, IMG_HEIGHT // 4),
        "bottom-left": (IMG_WIDTH // 4, 3 * IMG_HEIGHT // 4),
        "bottom-right": (3 * IMG_WIDTH // 4, 3 * IMG_HEIGHT // 4),
        "center": (IMG_WIDTH // 2, IMG_HEIGHT // 2),
    }
    cx, cy = positions_coords[position]
    size = 80
    
    # Draw shape
    if shape_name == "circle":
        draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=color_rgb, outline=(0, 0, 0), width=3)
    elif shape_name == "square":
        draw.rectangle([cx-size, cy-size, cx+size, cy+size], fill=color_rgb, outline=(0, 0, 0), width=3)
    elif shape_name == "triangle":
        points = [(cx, cy-size), (cx-size, cy+size), (cx+size, cy+size)]
        draw.polygon(points, fill=color_rgb, outline=(0, 0, 0), width=3)
    elif shape_name == "star":
        points = []
        for i in range(5):
            import math
            angle = math.pi/2 + i * 2 * math.pi / 5
            points.append((cx + size * math.cos(angle), cy - size * math.sin(angle)))
            angle += math.pi / 5
            points.append((cx + size/2 * math.cos(angle), cy - size/2 * math.sin(angle)))
        draw.polygon(points, fill=color_rgb, outline=(0, 0, 0), width=2)
    
    font = get_font(30)
    draw.text((cx - 50, cy + size + 10), "HIDDEN", fill=(100, 100, 100), font=font)
    
    blur_intensity = random.randint(18, 25)
    blurred_img = apply_extreme_blur(img, blur_intensity)
    
    # Question asks about what you will see AFTER sharpening
    question_type = random.choice(["shape", "color", "position"])
    
    if question_type == "shape":
        question = "There is a hidden shape in this blurred image. Use the image_unblur_tool with sharpen to reveal it. What shape is hidden?"
        answer = shape_name
    elif question_type == "color":
        question = "There is a hidden colored shape in this blurred image. Use the image_unblur_tool with sharpen to reveal it. What color is the hidden shape?"
        answer = color_name
    else:
        question = f"There is a hidden {color_name} {shape_name} in this blurred image. Use the image_unblur_tool with sharpen to reveal it. Where is it located (top-left, top-right, bottom-left, bottom-right, or center)?"
        answer = position

    expected_tool_calls = [{"name": "image_unblur_tool", "blur_type": "sharpen"}]
    
    metadata = {
        "task_type": "hidden_shape_reveal",
        "blur_intensity": blur_intensity,
        "shape": shape_name,
        "color": color_name,
        "position": position,
        "question_type": question_type
    }
    
    return blurred_img, question, answer, expected_tool_calls, metadata


def generate_number_equation_scene():
    """
    Generate a hidden equation. The model must sharpen to see and solve it.
    """
    bg_color = (255, 255, 255)
    
    # Generate simple equation
    num1 = random.randint(10, 99)
    num2 = random.randint(1, 20)
    op = random.choice(["+", "-"])
    
    if op == "+":
        result = num1 + num2
    else:
        result = num1 - num2
    
    equation = f"{num1} {op} {num2} = ?"
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    font = get_font(100)
    bbox = draw.textbbox((0, 0), equation, font=font)
    text_width = bbox[2] - bbox[0]
    x = (IMG_WIDTH - text_width) // 2
    y = IMG_HEIGHT // 2 - 50
    
    draw.text((x, y), equation, fill=(0, 0, 0), font=font)
    
    # Apply moderate blur (not too strong)
    blur_intensity = random.randint(8, 12)
    blurred_img = apply_extreme_blur(img, blur_intensity)
    
    question = "There is a hidden math equation in this blurred image. Use the image_unblur_tool with sharpen to reveal it. What is the answer to the equation?"
    answer = str(result)

    expected_tool_calls = [{"name": "image_unblur_tool", "blur_type": "sharpen"}]
    
    metadata = {
        "task_type": "hidden_equation",
        "blur_intensity": blur_intensity,
        "equation": equation,
        "result": result
    }
    
    return blurred_img, question, answer, expected_tool_calls, metadata


def generate_word_scene():
    """
    Generate a hidden word. The model must sharpen to read it.
    """
    bg_color = (255, 255, 255)
    
    # Use external mandatory word list (data/words.txt)
    word = random.choice(WORD_LIST)
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    font = get_font(120)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    x = (IMG_WIDTH - text_width) // 2
    y = IMG_HEIGHT // 2 - 60
    
    draw.text((x, y), word, fill=(0, 0, 0), font=font)
    
    # Apply moderate blur
    blur_intensity = random.randint(8, 12)
    blurred_img = apply_extreme_blur(img, blur_intensity)
    
    question = "There is a hidden word in this blurred image. Use the image_unblur_tool with sharpen to reveal it. What word is written?"
    answer = word

    expected_tool_calls = [{"name": "image_unblur_tool", "blur_type": "sharpen"}]
    
    metadata = {
        "task_type": "hidden_word",
        "blur_intensity": blur_intensity,
        "word": word
    }
    
    return blurred_img, question, answer, expected_tool_calls, metadata


def create_sample(idx, output_dir, scene_generators):
    """Create one sample with balanced scene types."""
    generator = scene_generators[idx % len(scene_generators)]
    
    img, question, answer, expected_tool_calls, metadata = generator()
    
    img_path = os.path.join(output_dir, "images", f"blur_reveal_{idx:05d}.png")
    img.save(img_path)
    img_abs = os.path.abspath(img_path)
    
    user_content = f"<image>\n{question}"
    
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    # tools_kwargs for multi-turn tooling
    tools_kwargs = {
        "image_unblur_tool": {
            "create_kwargs": {
                "image": img_abs
            }
        }
    }
    
    # reward_model must be a dict with ground_truth for validation
    reward_model = {
        "style": "rule",
        "ground_truth": answer,
        "expected_tool_calls": expected_tool_calls
    }
    
    return {
        "prompt": prompt,
        "answer": answer,
        "images": [{"image": img_abs}],
        "expected_tool_calls": expected_tool_calls,
        "metadata": metadata,
        "data_source": "blur",
        "reward_model": reward_model,
        "agent_name": "tool_agent",
        "tools_kwargs": tools_kwargs,
        "available_tools": ["image_unblur_tool"],
        "extra_info": {
            "expected_tool_calls": expected_tool_calls,
            "metadata": metadata,
        },
    }


def save_as_verl_parquet(rows, output_dir, split_name="train"):
    os.makedirs(output_dir, exist_ok=True)

    pa_rows = []
    for r in rows:
        pa_rows.append({
            "prompt": r["prompt"],
            "answer": r["answer"],
            "images": r["images"],
            "expected_tool_calls": r.get("expected_tool_calls", []),
            "metadata": r.get("metadata", {}),
            "data_source": r.get("data_source", "blur"),
            "reward_model": r.get("reward_model", {}),
            "agent_name": r.get("agent_name", "tool_agent"),
            "tools_kwargs": r.get("tools_kwargs", {}),
            "available_tools": r.get("available_tools", ["image_unblur_tool"]),
            "extra_info": {
                "expected_tool_calls": r.get("expected_tool_calls", []),
                "metadata": r.get("metadata", {}),
            },
        })

    table = pa.Table.from_pylist(pa_rows)

    out_path = os.path.join(output_dir, f"{split_name}.parquet")
    pq.write_table(table, out_path)
    return out_path, len(pa_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="blur_reveal")
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--test_samples", type=int, default=50)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    
    # Scene generators - equations and words (simple tasks)
    scene_generators = [
        generate_number_equation_scene,
        generate_word_scene,
    ]
    
    # Generate train samples
    print(f"Generating {args.train_samples} training samples...")
    train_data = []
    for idx in tqdm(range(args.train_samples)):
        sample = create_sample(idx, args.output_dir, scene_generators)
        train_data.append(sample)
    
    # Generate test samples
    print(f"Generating {args.test_samples} test samples...")
    test_data = []
    for idx in tqdm(range(args.test_samples)):
        sample = create_sample(args.train_samples + idx, args.output_dir, scene_generators)
        test_data.append(sample)
    
    print(f"Saving {len(train_data)} train and {len(test_data)} test examples to parquet...")
    train_path, train_count = save_as_verl_parquet(train_data, args.output_dir, split_name="train")
    test_path, test_count = save_as_verl_parquet(test_data, args.output_dir, split_name="test")

    print(f"  train_files: {train_path}")
    print(f"  val_files:   {test_path}")

    # Print task distribution using in-memory metadata
    counts = Counter([r.get("metadata", {}).get("task_type", "unknown") for r in train_data])
    print("Task distribution (train):")
    for t, c in counts.items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()