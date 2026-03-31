#!/usr/bin/env python3
"""
Generate zoom-task dataset for VERL training.
Mixed difficulty: easy tasks (visible without zoom) + hard tasks (require zoom).
The model must learn WHEN to use the image_zoom_in_tool.
"""

import os
import random
import argparse
import math
import string
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from collections import Counter


FONT_PATH = "arial.ttf"

# Image sizes: large enough that small details become invisible at ~448px processing
EASY_IMG_SIZE_RANGE = (800, 1200)
HARD_IMG_SIZE_RANGE = (1000, 1500)

# Target sizes
EASY_TARGET_SIZE_RANGE = (80, 150)  # Clearly visible at reduced resolution
HARD_TARGET_SIZE_RANGE = (12, 25)   # Indistinguishable at reduced resolution

BACKGROUND_COLORS = [
    (240, 240, 240), (255, 255, 255), (220, 220, 220),
    (245, 245, 220), (230, 230, 250), (255, 250, 240),
    (240, 255, 240), (255, 240, 245), (240, 248, 255),
]

# Distinct colors (easily named, hard to confuse at full resolution)
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 180, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "orange": (255, 140, 0),
    "purple": (128, 0, 200),
    "brown": (139, 69, 19),
    "pink": (255, 105, 180),
}

# Colors that look similar at low resolution (confusers for hard tasks)
CONFUSABLE_GROUPS = [
    ["red", "orange", "brown"],
    ["blue", "purple", "cyan"],
    ["green", "cyan"],
    ["yellow", "orange"],
    ["magenta", "pink", "red"],
]

SHAPES = ['circle', 'square', 'triangle', 'diamond', 'star', 'cross']

# Digits for tiny digit reading tasks
DIGITS = list("0123456789")

# Short words for fine text reading tasks
SHORT_WORDS = [
    "apple", "maple", "table", "cable", "fable",
    "stone", "store", "stove", "score", "snore",
    "plate", "place", "plane", "plant", "plain",
    "light", "night", "right", "sight", "tight",
    "beach", "peach", "reach", "teach", "leach",
]

# Region descriptions for questions (no coordinates)
REGION_NAMES = {
    "top-left": (0.0, 0.0, 0.4, 0.4),
    "top-center": (0.3, 0.0, 0.7, 0.4),
    "top-right": (0.6, 0.0, 1.0, 0.4),
    "center-left": (0.0, 0.3, 0.4, 0.7),
    "center": (0.3, 0.3, 0.7, 0.7),
    "center-right": (0.6, 0.3, 1.0, 0.7),
    "bottom-left": (0.0, 0.6, 0.4, 1.0),
    "bottom-center": (0.3, 0.6, 0.7, 1.0),
    "bottom-right": (0.6, 0.6, 1.0, 1.0),
}

SYSTEM_PROMPT = """You are a helpful assistant. You have access to visual tools that can help you examine images more closely. Use them when you need to see fine details that might not be visible at the current resolution. Provide your final answer in <boxed></boxed> tags."""


def random_bg():
    return random.choice(BACKGROUND_COLORS)


def get_region_name(x, y, img_w, img_h):
    """Get the natural language region name for a coordinate."""
    rx, ry = x / img_w, y / img_h
    best_name = "center"
    best_dist = float("inf")
    for name, (x1, y1, x2, y2) in REGION_NAMES.items():
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = (rx - cx) ** 2 + (ry - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def place_in_region(region_name, img_w, img_h, obj_size, margin=20):
    """Place an object within a named region, return (x, y)."""
    x1_frac, y1_frac, x2_frac, y2_frac = REGION_NAMES[region_name]
    x_min = max(margin, int(x1_frac * img_w))
    x_max = max(x_min + 1, int(x2_frac * img_w) - obj_size - margin)
    y_min = max(margin, int(y1_frac * img_h))
    y_max = max(y_min + 1, int(y2_frac * img_h) - obj_size - margin)
    return random.randint(x_min, x_max), random.randint(y_min, y_max)


def draw_shape(draw, shape, x, y, size, color):
    """Draw a shape at (x, y) with given size and color."""
    if shape == 'circle':
        draw.ellipse([x, y, x + size, y + size], fill=color, outline=(0, 0, 0), width=max(1, size // 20))
    elif shape == 'square':
        draw.rectangle([x, y, x + size, y + size], fill=color, outline=(0, 0, 0), width=max(1, size // 20))
    elif shape == 'triangle':
        points = [(x + size // 2, y), (x, y + size), (x + size, y + size)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=max(1, size // 20))
    elif shape == 'diamond':
        cx, cy = x + size // 2, y + size // 2
        points = [(cx, y), (x + size, cy), (cx, y + size), (x, cy)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=max(1, size // 20))
    elif shape == 'star':
        cx, cy = x + size // 2, y + size // 2
        outer_r, inner_r = size // 2, size // 4
        points = []
        for k in range(10):
            r = outer_r if k % 2 == 0 else inner_r
            angle = math.pi / 2 + k * math.pi / 5
            points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=max(1, size // 20))
    elif shape == 'cross':
        third = size // 3
        points = [
            (x + third, y), (x + 2 * third, y),
            (x + 2 * third, y + third), (x + size, y + third),
            (x + size, y + 2 * third), (x + 2 * third, y + 2 * third),
            (x + 2 * third, y + size), (x + third, y + size),
            (x + third, y + 2 * third), (x, y + 2 * third),
            (x, y + third), (x + third, y + third),
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=max(1, size // 20))
    return [x, y, x + size, y + size]


def add_visual_clutter(draw, img_w, img_h, num_elements=15):
    """Add random visual noise to make the scene more complex."""
    for _ in range(num_elements):
        color_name = random.choice(list(COLORS.keys()))
        color = COLORS[color_name]
        size = random.randint(20, 60)
        x = random.randint(0, img_w - size)
        y = random.randint(0, img_h - size)
        shape = random.choice(SHAPES)
        # Draw with some transparency effect (lighter color)
        lighter = tuple(min(255, c + 80) for c in color)
        draw_shape(draw, shape, x, y, size, lighter)


def generate_easy_task():
    """Generate a task with large, clearly visible targets."""
    img_w = random.randint(*EASY_IMG_SIZE_RANGE)
    img_h = random.randint(*EASY_IMG_SIZE_RANGE)
    bg_color = random_bg()
    img = Image.new('RGB', (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Add some clutter
    add_visual_clutter(draw, img_w, img_h, num_elements=10)

    # Place the target — large and clearly visible
    target_size = random.randint(*EASY_TARGET_SIZE_RANGE)
    target_region = random.choice(list(REGION_NAMES.keys()))
    tx, ty = place_in_region(target_region, img_w, img_h, target_size)

    task_type = random.choice(["shape_color", "shape_type"])

    target_color_name = random.choice(list(COLORS.keys()))
    target_color = COLORS[target_color_name]
    target_shape = random.choice(SHAPES)

    # Draw the target
    draw_shape(draw, target_shape, tx, ty, target_size, target_color)

    # Draw a visible label near the target
    try:
        label_font = ImageFont.truetype(FONT_PATH, max(16, target_size // 4))
    except Exception:
        label_font = ImageFont.load_default()
    label = random.choice(string.ascii_uppercase) + str(random.randint(1, 9))
    draw.text((tx, max(0, ty - 20)), label, fill=(0, 0, 0), font=label_font)

    if task_type == "shape_color":
        question = random.choice([
            f"Look at the shape labeled '{label}' in the {target_region} area of the image. What color is it?",
            f"Find '{label}' in the {target_region} region. What color is this shape?",
            f"There is a shape marked '{label}' in the {target_region} part of the image. What is its color?",
        ])
        answer = f"<boxed>{target_color_name}</boxed>"
    else:
        question = random.choice([
            f"Look at the shape labeled '{label}' in the {target_region} area of the image. What shape is it?",
            f"Find '{label}' in the {target_region} region. What type of shape is it?",
            f"There is a shape marked '{label}' in the {target_region} part of the image. What shape is it?",
        ])
        answer = f"<boxed>{target_shape}</boxed>"

    question += " Put your final answer in <boxed></boxed> tags."

    bbox = [tx, ty, tx + target_size, ty + target_size]
    metadata = {
        "difficulty": "easy",
        "task_type": task_type,
        "target_region": target_region,
        "target_size": target_size,
        "target_shape": target_shape,
        "target_color": target_color_name,
        "target_bbox": bbox,
        "image_size": [img_w, img_h],
    }

    return img, question, answer, metadata


def generate_hard_shape_task():
    """Generate a task with tiny shapes that require zoom to identify."""
    img_w = random.randint(*HARD_IMG_SIZE_RANGE)
    img_h = random.randint(*HARD_IMG_SIZE_RANGE)
    bg_color = random_bg()
    img = Image.new('RGB', (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Add clutter
    add_visual_clutter(draw, img_w, img_h, num_elements=20)

    # Choose a target region
    target_region = random.choice(list(REGION_NAMES.keys()))

    # Place the target — tiny
    target_size = random.randint(*HARD_TARGET_SIZE_RANGE)
    tx, ty = place_in_region(target_region, img_w, img_h, target_size)

    # Pick confusable color group for harder discrimination
    confuse_group = random.choice(CONFUSABLE_GROUPS)
    target_color_name = random.choice(confuse_group)
    target_color = COLORS[target_color_name]
    target_shape = random.choice(SHAPES)

    # Draw the target
    draw_shape(draw, target_shape, tx, ty, target_size, target_color)

    # Place 3-5 confuser shapes nearby (same region, different properties)
    num_confusers = random.randint(3, 5)
    for _ in range(num_confusers):
        c_size = random.randint(*HARD_TARGET_SIZE_RANGE)
        try:
            cx, cy = place_in_region(target_region, img_w, img_h, c_size)
        except Exception:
            continue
        # Different color from target but from same confusable group
        c_color_name = random.choice([c for c in confuse_group if c != target_color_name] or list(COLORS.keys()))
        c_color = COLORS[c_color_name]
        c_shape = random.choice([s for s in SHAPES if s != target_shape] or SHAPES)
        draw_shape(draw, c_shape, cx, cy, c_size, c_color)

    # Draw a small but readable marker near the target
    try:
        marker_font = ImageFont.truetype(FONT_PATH, max(10, target_size))
    except Exception:
        marker_font = ImageFont.load_default()

    marker = random.choice(["*", "+", "x", "o", "#"])
    draw.text((tx - 5, max(0, ty - target_size - 2)), marker, fill=(0, 0, 0), font=marker_font)

    task_type = random.choice(["shape_color", "shape_type"])

    if task_type == "shape_color":
        question = random.choice([
            f"In the {target_region} area of the image, there is a tiny shape near the '{marker}' mark. What color is it? You may need to zoom in to see it clearly.",
            f"Look at the small shape marked with '{marker}' in the {target_region} region. What color is this shape?",
            f"There is a very small shape near the '{marker}' symbol in the {target_region} part of the image. What is its color?",
        ])
        answer = f"<boxed>{target_color_name}</boxed>"
    else:
        question = random.choice([
            f"In the {target_region} area of the image, there is a tiny shape near the '{marker}' mark. What type of shape is it? You may need to zoom in to see it clearly.",
            f"Look at the small shape marked with '{marker}' in the {target_region} region. What shape is it?",
            f"There is a very small shape near the '{marker}' symbol in the {target_region} part of the image. What shape is it?",
        ])
        answer = f"<boxed>{target_shape}</boxed>"

    question += " Put your final answer in <boxed></boxed> tags."

    bbox = [tx, ty, tx + target_size, ty + target_size]
    metadata = {
        "difficulty": "hard",
        "task_type": task_type,
        "target_region": target_region,
        "target_size": target_size,
        "target_shape": target_shape,
        "target_color": target_color_name,
        "target_bbox": bbox,
        "image_size": [img_w, img_h],
        "num_confusers": num_confusers,
    }

    return img, question, answer, metadata


def generate_hard_digit_task():
    """Generate a task with tiny digits that require zoom to read."""
    img_w = random.randint(*HARD_IMG_SIZE_RANGE)
    img_h = random.randint(*HARD_IMG_SIZE_RANGE)
    bg_color = random_bg()
    img = Image.new('RGB', (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Add clutter
    add_visual_clutter(draw, img_w, img_h, num_elements=20)

    target_region = random.choice(list(REGION_NAMES.keys()))

    # Draw target digit — very small font
    font_size = random.randint(12, 16)
    try:
        digit_font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        digit_font = ImageFont.load_default()

    target_digit = random.choice(DIGITS)
    tx, ty = place_in_region(target_region, img_w, img_h, font_size)

    color_name = random.choice(list(COLORS.keys()))
    color = COLORS[color_name]
    draw.text((tx, ty), target_digit, fill=color, font=digit_font)

    # Place confuser digits nearby
    num_confusers = random.randint(3, 6)
    confuser_digits = [d for d in DIGITS if d != target_digit]
    for _ in range(num_confusers):
        cd = random.choice(confuser_digits)
        try:
            cx, cy = place_in_region(target_region, img_w, img_h, font_size)
        except Exception:
            continue
        c_color = COLORS[random.choice(list(COLORS.keys()))]
        draw.text((cx, cy), cd, fill=c_color, font=digit_font)

    # Add a reference marker
    marker = random.choice(["=>", "->", ">>"])
    try:
        marker_font = ImageFont.truetype(FONT_PATH, max(10, font_size))
    except Exception:
        marker_font = ImageFont.load_default()
    draw.text((max(0, tx - 25), ty), marker, fill=(0, 0, 0), font=marker_font)

    question = random.choice([
        f"In the {target_region} area, there is a small {color_name} digit near the '{marker}' arrow. What digit is it? You may need to zoom in.",
        f"Look at the tiny {color_name} number near '{marker}' in the {target_region} region. What digit is written there?",
        f"There is a small {color_name} digit marked with '{marker}' in the {target_region} part. What is the digit?",
    ])
    question += " Put your final answer in <boxed></boxed> tags."

    answer = f"<boxed>{target_digit}</boxed>"

    bbox = draw.textbbox((tx, ty), target_digit, font=digit_font)
    metadata = {
        "difficulty": "hard",
        "task_type": "digit_reading",
        "target_region": target_region,
        "target_digit": target_digit,
        "target_color": color_name,
        "font_size": font_size,
        "target_bbox": list(bbox),
        "image_size": [img_w, img_h],
        "num_confusers": num_confusers,
    }

    return img, question, answer, metadata


def generate_hard_text_task():
    """Generate a task with tiny text that requires zoom to read."""
    img_w = random.randint(*HARD_IMG_SIZE_RANGE)
    img_h = random.randint(*HARD_IMG_SIZE_RANGE)
    bg_color = random_bg()
    img = Image.new('RGB', (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Add clutter
    add_visual_clutter(draw, img_w, img_h, num_elements=20)

    target_region = random.choice(list(REGION_NAMES.keys()))

    # Small font for the text
    font_size = random.randint(12, 16)
    try:
        text_font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        text_font = ImageFont.load_default()

    # Pick a word group where words look similar at low res
    word_group_idx = random.randint(0, len(SHORT_WORDS) // 5 - 1)
    word_group = SHORT_WORDS[word_group_idx * 5:(word_group_idx + 1) * 5]
    target_word = random.choice(word_group)

    tx, ty = place_in_region(target_region, img_w, img_h, font_size * 4)

    color_name = random.choice(list(COLORS.keys()))
    color = COLORS[color_name]
    draw.text((tx, ty), target_word, fill=color, font=text_font)

    # Place confuser words nearby
    confuser_words = [w for w in word_group if w != target_word]
    for i, cw in enumerate(confuser_words[:3]):
        try:
            cx, cy = place_in_region(target_region, img_w, img_h, font_size * 4)
        except Exception:
            continue
        c_color = COLORS[random.choice(list(COLORS.keys()))]
        draw.text((cx, cy), cw, fill=c_color, font=text_font)

    # Reference marker
    marker_char = random.choice(["[*]", "[!]", "[?]"])
    try:
        marker_font = ImageFont.truetype(FONT_PATH, max(10, font_size))
    except Exception:
        marker_font = ImageFont.load_default()
    draw.text((max(0, tx - 30), ty), marker_char, fill=(0, 0, 0), font=marker_font)

    question = random.choice([
        f"In the {target_region} area, there is a small {color_name} word near the '{marker_char}' marker. What word is written there? You may need to zoom in to read it.",
        f"Look at the tiny {color_name} text near '{marker_char}' in the {target_region} region. What word is it?",
        f"There is small {color_name} text marked with '{marker_char}' in the {target_region} part. What does it say?",
    ])
    question += " Put your final answer in <boxed></boxed> tags."

    answer = f"<boxed>{target_word}</boxed>"

    bbox = draw.textbbox((tx, ty), target_word, font=text_font)
    metadata = {
        "difficulty": "hard",
        "task_type": "text_reading",
        "target_region": target_region,
        "target_word": target_word,
        "target_color": color_name,
        "font_size": font_size,
        "target_bbox": list(bbox),
        "image_size": [img_w, img_h],
    }

    return img, question, answer, metadata


def generate_sample(easy_ratio=0.4):
    """Generate a single sample, choosing difficulty tier."""
    if random.random() < easy_ratio:
        return generate_easy_task()
    else:
        # Randomly pick among hard task types
        hard_generator = random.choice([
            generate_hard_shape_task,
            generate_hard_digit_task,
            generate_hard_text_task,
        ])
        return hard_generator()


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
            "data_source": r.get("data_source", "zoom"),
            "reward_model": r.get("reward_model", {}),
            "agent_name": r.get("agent_name", "tool_agent"),
            "available_tools": r.get("available_tools", ["image_zoom_in_tool"]),
            "extra_info": {
                "tools_kwargs": r.get("tools_kwargs", {}),
                "metadata": r.get("metadata", {}),
            },
        })

    table = pa.Table.from_pylist(pa_rows)

    out_path = os.path.join(output_dir, f"{split_name}.parquet")
    pq.write_table(table, out_path)
    return out_path, len(pa_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate zoom-task dataset for VERL training")
    parser.add_argument("--output_dir", type=str, default="zoom_open", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=300, help="Total samples (train+test)")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction for train split")
    parser.add_argument("--easy_ratio", type=float, default=0.4, help="Fraction of easy tasks")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Generating {args.num_samples} samples (easy_ratio={args.easy_ratio})...")
    rows = []
    difficulty_counts = Counter()

    for i in tqdm(range(args.num_samples), desc="Generating"):
        img, question, answer, metadata = generate_sample(easy_ratio=args.easy_ratio)
        difficulty_counts[metadata["difficulty"]] += 1

        img_name = f"zoom_{i:05d}.png"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        img_abs = os.path.abspath(img_path)

        user_content = f"<image>\n{question}"

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tools_kwargs = {
            "image_zoom_in_tool": {
                "create_kwargs": {"image": img_abs}
            }
        }

        reward_model = {
            "style": "rule",
            "ground_truth": answer,
        }

        row = {
            "prompt": prompt,
            "answer": answer,
            "images": [img_abs],
            "expected_tool_calls": [],
            "metadata": metadata,
            "data_source": "zoom",
            "reward_model": reward_model,
            "agent_name": "tool_agent",
            "tools_kwargs": tools_kwargs,
            "available_tools": ["image_zoom_in_tool"],
        }
        rows.append(row)

    # Shuffle to mix easy and hard
    random.shuffle(rows)

    train_n = int(args.train_frac * len(rows))
    train_rows = rows[:train_n]
    test_rows = rows[train_n:]

    print(f"\nDifficulty distribution: {dict(difficulty_counts)}")
    print(f"Saving {len(train_rows)} train and {len(test_rows)} test examples...")

    train_path, train_count = save_as_verl_parquet(train_rows, args.output_dir, split_name="train")
    test_path, test_count = save_as_verl_parquet(test_rows, args.output_dir, split_name="test")

    # Verification
    try:
        df = pd.read_parquet(train_path, engine="pyarrow")
        print(f"\nColumns: {df.columns.tolist()}")
        sample = df.iloc[0]
        print(f"data_source: {sample['data_source']}")
        print(f"available_tools: {sample['available_tools']}")
        print(f"reward_model ground_truth: {sample['reward_model'].get('ground_truth', 'N/A')}")

        difficulties = [m.get("difficulty", "unknown") for m in df["metadata"]]
        task_types = [m.get("task_type", "unknown") for m in df["metadata"]]
        print(f"Train difficulty distribution: {dict(Counter(difficulties))}")
        print(f"Train task type distribution: {dict(Counter(task_types))}")
    except Exception as e:
        print(f"Verification failed: {e}")

    print(f"\n  train_files: {os.path.abspath(train_path)}")
    print(f"  val_files:   {os.path.abspath(test_path)}")


if __name__ == "__main__":
    main()
