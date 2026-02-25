#!/usr/bin/env python3
"""
Generate bbox dataset for VERL training.
Shapes have random labels. Questions identify by label, ask about color or shape.
The answer is NEVER in the question.
"""

import os
import random
import argparse
import json
import math
import string
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from collections import Counter


FONT_PATH = "arial.ttf"
IMG_WIDTH_RANGE = (400, 1200)
IMG_HEIGHT_RANGE = (300, 900)

BACKGROUND_COLORS = [
    (240, 240, 240), (255, 255, 255), (220, 220, 220),
    (245, 245, 220), (230, 230, 250), (255, 250, 240),
    (240, 255, 240), (255, 240, 245), (240, 248, 255),
]
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (255, 99, 71), (50, 205, 50), (30, 144, 255),
    (255, 215, 0), (219, 112, 147), (0, 206, 209),
    (255, 69, 0), (154, 205, 50), (138, 43, 226),
]
COLOR_NAMES = {
    (255, 0, 0): "red", (0, 255, 0): "green", (0, 0, 255): "blue",
    (255, 255, 0): "yellow", (255, 0, 255): "magenta", (0, 255, 255): "cyan",
    (255, 128, 0): "orange", (128, 0, 255): "purple", (0, 255, 128): "lime",
    (128, 128, 0): "olive", (0, 128, 128): "teal", (128, 0, 128): "violet",
    (255, 99, 71): "tomato", (50, 205, 50): "limegreen", (30, 144, 255): "dodgerblue",
    (255, 215, 0): "gold", (219, 112, 147): "palevioletred", (0, 206, 209): "darkturquoise",
    (255, 69, 0): "orangered", (154, 205, 50): "yellowgreen", (138, 43, 226): "blueviolet",
}

SHAPES = ['circle', 'square', 'triangle', 'diamond', 'pentagon', 'star', 'heart', 'arrow', 'cross', 'ring']

def random_dims():
    return random.randint(*IMG_WIDTH_RANGE), random.randint(*IMG_HEIGHT_RANGE)

def random_bg():
    return random.choice(BACKGROUND_COLORS)

def rgb_to_color_name(rgb):
    return COLOR_NAMES.get(tuple(rgb), "unknown")

def generate_label():
    """Generate a random label like A1, B2, X7, etc."""
    letter = random.choice(string.ascii_uppercase)
    number = random.randint(1, 9)
    return f"{letter}{number}"

SYSTEM_PROMPT = """You are a helpful assistant that analyzes images using the bounding box tool.

When asked about specific regions or objects in an image:
1. Use image_bbox_tool to draw a bounding box around the target region
2. After receiving the tool response, provide your final answer in <boxed></boxed> tags

Always use the bounding box tool before answering questions about image regions."""



def draw_shape(draw, shape, x, y, size, color):
    """Draw a shape and return its bounding box."""
    if shape == 'circle':
        draw.ellipse([x, y, x + size, y + size], fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'square':
        draw.rectangle([x, y, x + size, y + size], fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'triangle':
        points = [(x + size // 2, y), (x, y + size), (x + size, y + size)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'diamond':
        cx, cy = x + size // 2, y + size // 2
        points = [(cx, y), (x + size, cy), (cx, y + size), (x, cy)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'pentagon':
        cx, cy = x + size // 2, y + size // 2
        r = size // 2
        points = [(cx + r * math.sin(2 * math.pi * k / 5), cy - r * math.cos(2 * math.pi * k / 5)) for k in range(5)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'star':
        cx, cy = x + size // 2, y + size // 2
        outer_r, inner_r = size // 2, size // 4
        points = []
        for k in range(10):
            r = outer_r if k % 2 == 0 else inner_r
            angle = math.pi / 2 + k * math.pi / 5
            points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'heart':
        cx, cy = x + size // 2, y + size // 2
        points = []
        for t in range(100):
            angle = 2 * math.pi * t / 100
            hx = 16 * (math.sin(angle) ** 3)
            hy = 13 * math.cos(angle) - 5 * math.cos(2 * angle) - 2 * math.cos(3 * angle) - math.cos(4 * angle)
            points.append((cx + hx * size / 40, cy - hy * size / 40))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'arrow':
        points = [
            (x, y + size // 3),
            (x + size * 2 // 3, y + size // 3),
            (x + size * 2 // 3, y),
            (x + size, y + size // 2),
            (x + size * 2 // 3, y + size),
            (x + size * 2 // 3, y + size * 2 // 3),
            (x, y + size * 2 // 3),
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
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
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'ring':
        draw.ellipse([x, y, x + size, y + size], fill=color, outline=(0, 0, 0), width=2)
        inner = size // 3
        draw.ellipse([x + inner, y + inner, x + size - inner, y + size - inner], fill=random_bg(), outline=(0, 0, 0), width=1)
    
    return [x, y, x + size, y + size]


def load_words():
    """Load words from file, with fallback."""
    try:
        with open("words.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return ["apple", "banana", "cherry", "dog", "elephant", "flower", "guitar", "house", "island", "jungle"]

WORDS = load_words()


def generate_mixed_scene():
    """
    Generate a scene with labeled shapes and text.
    Each shape gets a random label (A1, B2, etc.) drawn on it.
    Questions identify by label, ask about color OR shape (never both in question).
    """
    img_w, img_h = random_dims()
    bg_color = random_bg()
    img = Image.new('RGB', (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    all_targets = []
    used_labels = set()
    
    try:
        label_font = ImageFont.truetype(FONT_PATH, 14)
        text_font = ImageFont.truetype(FONT_PATH, random.randint(18, 28))
    except:
        label_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    
    colors_list = list(COLORS)
    random.shuffle(colors_list)
    
    num_shapes = random.randint(4, 8)
    
    for i in range(num_shapes):
        for _ in range(50):
            label = generate_label()
            if label not in used_labels:
                used_labels.add(label)
                break
        
        shape = random.choice(SHAPES)
        color = colors_list[i % len(colors_list)]
        size = random.randint(50, 100)
        x = random.randint(20, max(30, img_w - size - 20))
        y = random.randint(20, max(30, img_h - size - 20))
        
        bbox = draw_shape(draw, shape, x, y, size, color)
        
        # Draw the label on the shape (centered, in contrasting color)
        label_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        label_bbox = draw.textbbox((0, 0), label, font=label_font)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        label_x = x + (size - label_w) // 2
        label_y = y + (size - label_h) // 2
        draw.text((label_x, label_y), label, fill=label_color, font=label_font)
        
        all_targets.append({
            "type": "shape",
            "label": label,
            "shape": shape,
            "color": color,
            "color_name": rgb_to_color_name(color),
            "bbox": bbox,
        })
    
    num_texts = random.randint(2, 4)
    words_sample = random.sample(WORDS, min(num_texts, len(WORDS)))
    
    for i, word in enumerate(words_sample):
        color = colors_list[(num_shapes + i) % len(colors_list)]
        x = random.randint(30, max(40, img_w - 150))
        y = random.randint(30, max(40, img_h - 50))
        draw.text((x, y), word, fill=color, font=text_font)
        bbox = draw.textbbox((x, y), word, font=text_font)
        
        all_targets.append({
            "type": "text",
            "label": word,  # The word itself is the label
            "word": word,
            "color": color,
            "color_name": rgb_to_color_name(color),
            "bbox": list(bbox),
        })
    
    # Pick a random target
    target = random.choice(all_targets)
    
    if target["type"] == "shape":
        # Question identifies by label, asks about color OR shape
        ask_type = random.choice(["color", "shape"])
        
        if ask_type == "color":
            question_templates = [
                f"Find the shape labeled '{target['label']}' and draw a bounding box around it. What color is it?",
                f"Use bbox tool to locate the shape with label '{target['label']}'. What is its color?",
                f"Draw a bounding box around '{target['label']}'. Tell me what color this shape is.",
            ]
            answer = f"<boxed>{target['color_name']}</boxed>"
        else:  # ask_type == "shape"
            question_templates = [
                f"Find the shape labeled '{target['label']}' and draw a bounding box around it. What shape is it?",
                f"Use bbox tool to locate the shape with label '{target['label']}'. What type of shape is this?",
                f"Draw a bounding box around '{target['label']}'. Tell me what shape this is.",
            ]
            answer = f"<boxed>{target['shape']}</boxed>"
        
        ref_label = target['label']
    else:
        # Text: identify by word, ask about color
        question_templates = [
            f"Find the word '{target['word']}' and draw a bounding box around it. What color is it?",
            f"Use bbox tool to locate the text '{target['word']}'. What color is this word written in?",
            f"Draw a bounding box around the word '{target['word']}'. What is its color?",
        ]
        answer = f"<boxed>{target['color_name']}</boxed>"
        ref_label = target['word']
    
    problem = random.choice(question_templates) + " Put your final answer in <boxed></boxed> tags."
    
    expected_tool_calls = [{
        "tool": "image_bbox_tool",
        "parameters": {"bbox_2d": target["bbox"], "label": ref_label}
    }]
    
    metadata = {
        "all_targets": all_targets,
        "target": target,
        "ground_truth_bbox": target["bbox"],
        "task_type": "mixed_scene"
    }
    
    return img, problem, answer, expected_tool_calls, metadata


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
            "data_source": r.get("data_source", "bbox"),
            "reward_model": r.get("reward_model", {}),
            "agent_name": r.get("agent_name", "tool_agent"),
            "tools_kwargs": r.get("tools_kwargs", {}),
            "available_tools": r.get("available_tools", ["image_bbox_tool"]),
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
    parser = argparse.ArgumentParser(description="Generate bbox/zoom dataset and save as VERL parquet (fixed)")
    parser.add_argument("--output_dir", type=str, default="bbox", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=200, help="Total samples to generate (train+test)")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction for train split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Generating {args.num_samples} samples...")
    rows = []
    for i in tqdm(range(args.num_samples), desc="Generating"):
        img, problem, answer, expected_tool_calls, metadata = generate_mixed_scene()

        img_name = f"bbox_{i:05d}.png"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        img_abs = os.path.abspath(img_path)

        user_content = (
            "<image>\n"
            f"{problem}\n\n"
            "Use the image_bbox_tool to mark the relevant region, then provide your answer."
        )
        
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tools_kwargs = {
            "image_bbox_tool": {
                "create_kwargs": {"image": img_abs}
            }
        }

        reward_model = {
            "style": "rule",
            "ground_truth": answer,
            "expected_tool_calls": expected_tool_calls
        }

        row = {
            "prompt": prompt,
            "answer": answer,
            "images": [{"image": img_abs}],
            "expected_tool_calls": expected_tool_calls,
            "metadata": {**metadata, "tools_kwargs": tools_kwargs},
            "data_source": "bbox",
            "reward_model": reward_model,
            "agent_name": "tool_agent",
            "tools_kwargs": tools_kwargs,
            "available_tools": ["image_bbox_tool"],
        }
        rows.append(row)

    train_n = int(args.train_frac * len(rows))
    train_rows = rows[:train_n]
    test_rows = rows[train_n:]

    print(f"Saving {len(train_rows)} train and {len(test_rows)} test examples to parquet...")
    train_path, train_count = save_as_verl_parquet(train_rows, args.output_dir, split_name="train")
    test_path, test_count = save_as_verl_parquet(test_rows, args.output_dir, split_name="test")

    try:
        df = pd.read_parquet(train_path, engine="pyarrow")
        print("Columns:", df.columns.tolist())
        sample = df.iloc[0]
        print("Type reward_model (python):", type(sample["reward_model"]))
        print("Type expected_tool_calls (python):", type(sample["expected_tool_calls"]))
        print("Type metadata (python):", type(sample["metadata"]))
        try:
            print("reward_model.get('ground_truth') ->", sample["reward_model"].get("ground_truth"))
        except Exception as e:
            print("ERROR accessing reward_model.get('ground_truth'):", e)

        counts = Counter([sample_meta.get("task_type", "unknown") for sample_meta in df["metadata"]])
        print("Task distribution (train):")
        for t, c in counts.items():
            print(f"  {t}: {c}")
    except Exception as e:
        print("Verification failed:", e)
    print(f"  train_files: {train_path}")
    print(f"  val_files:   {test_path}")


if __name__ == "__main__":
    main()
