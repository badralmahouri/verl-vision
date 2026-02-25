#!/usr/bin/env python3
"""
Generate flip images + VERL-compatible parquet dataset for image_flip_tool training.
Creates tasks where the model must use the flip tool to answer questions about flipped images.
"""

import os
import random
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from collections import Counter


FONT_PATH = "arial.ttf"
IMG_WIDTH = 800
IMG_HEIGHT = 600

BACKGROUND_COLOR = (240, 240, 240)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 255, 128)
]

SYSTEM_PROMPT = """You are a helpful assistant that analyzes images using tools.

When asked about flipped or mirrored versions of images:
1. Use the image_flip_tool to flip the image in the specified direction
2. After receiving the flipped image, analyze it and provide your final answer in <boxed></boxed> tags

Always use tools before answering questions about flipped images."""

def rgb_to_color_name(rgb):
    color_map = {
        (255, 0, 0): "red",
        (0, 255, 0): "green",
        (0, 0, 255): "blue",
        (255, 255, 0): "yellow",
        (255, 0, 255): "magenta",
        (0, 255, 255): "cyan",
        (255, 128, 0): "orange",
        (128, 0, 255): "purple",
        (0, 255, 128): "lime",
    }
    return color_map.get(tuple(rgb), "unknown")


def generate_asymmetric_arrow_scene():
    """Generate scene with an arrow pointing in one direction."""
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    arrow_color = random.choice(COLORS)
    arrow_x = IMG_WIDTH // 4
    arrow_y = IMG_HEIGHT // 2
    arrow_length = 300
    arrow_head_size = 50

    draw.rectangle([arrow_x, arrow_y - 20, arrow_x + arrow_length - arrow_head_size, arrow_y + 20], 
                   fill=arrow_color, outline=(0, 0, 0), width=2)
    
    arrow_tip_x = arrow_x + arrow_length
    points = [
        (arrow_x + arrow_length - arrow_head_size, arrow_y - 40),
        (arrow_tip_x, arrow_y),
        (arrow_x + arrow_length - arrow_head_size, arrow_y + 40)
    ]
    draw.polygon(points, fill=arrow_color, outline=(0, 0, 0), width=2)

    try:
        font = ImageFont.truetype(FONT_PATH, 32)
    except Exception:
        font = ImageFont.load_default()
    draw.text((50, 50), "START", fill=(0, 0, 0), font=font)

    direction = random.choice(["horizontal", "vertical"])
    
    if direction == "horizontal":
        # After horizontal flip, arrow points LEFT and START label is on RIGHT
        problem = f"If you flip this image horizontally, which direction will the arrow point? Put your final answer in <boxed></boxed> tags."
        answer = "<boxed>left</boxed>"
    else:
        # After vertical flip, arrow still points right but START label moves to bottom
        problem = f"If you flip this image vertically, will the 'START' label be at the top or bottom? Put your final answer in <boxed></boxed> tags."
        answer = "<boxed>bottom</boxed>"

    expected_tool_calls = [
        {
            "tool": "image_flip_tool",
            "parameters": {
                "direction": direction
            }
        }
    ]

    metadata = {
        "arrow_color": rgb_to_color_name(arrow_color),
        "flip_direction": direction,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "flip_arrow"
    }

    return img, problem, answer, expected_tool_calls, metadata


def generate_labeled_quadrants_scene():
    """Generate scene with labeled quadrants (A, B, C, D)."""
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw quadrant lines
    draw.line([(IMG_WIDTH // 2, 0), (IMG_WIDTH // 2, IMG_HEIGHT)], fill=(0, 0, 0), width=3)
    draw.line([(0, IMG_HEIGHT // 2), (IMG_WIDTH, IMG_HEIGHT // 2)], fill=(0, 0, 0), width=3)

    # Label quadrants with colors
    quadrant_colors = random.sample(COLORS, 4)
    quadrants = [
        {"name": "A", "pos": (IMG_WIDTH // 4, IMG_HEIGHT // 4), "color": quadrant_colors[0], "position": "top-left"},
        {"name": "B", "pos": (3 * IMG_WIDTH // 4, IMG_HEIGHT // 4), "color": quadrant_colors[1], "position": "top-right"},
        {"name": "C", "pos": (IMG_WIDTH // 4, 3 * IMG_HEIGHT // 4), "color": quadrant_colors[2], "position": "bottom-left"},
        {"name": "D", "pos": (3 * IMG_WIDTH // 4, 3 * IMG_HEIGHT // 4), "color": quadrant_colors[3], "position": "bottom-right"},
    ]

    try:
        font = ImageFont.truetype(FONT_PATH, 72)
    except Exception:
        font = ImageFont.load_default()

    for q in quadrants:
        # Draw colored circle
        cx, cy = q["pos"]
        radius = 60
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], 
                     fill=q["color"], outline=(0, 0, 0), width=2)
        # Draw label
        draw.text((cx - 20, cy - 40), q["name"], fill=(0, 0, 0), font=font)

    direction = random.choice(["horizontal", "vertical"])
    target_quadrant = random.choice(quadrants)
    
    if direction == "horizontal":
        # Horizontal flip swaps left-right: A<->B, C<->D
        swap_map = {"A": "B", "B": "A", "C": "D", "D": "C"}
        new_position = target_quadrant["position"].replace("left", "TEMP").replace("right", "left").replace("TEMP", "right")
        problem = f"After flipping this image horizontally, what letter will be in the {new_position} quadrant? Put your final answer in <boxed></boxed> tags."
        answer = f"<boxed>{target_quadrant['name']}</boxed>"
    else:
        # Vertical flip swaps top-bottom: A<->C, B<->D
        swap_map = {"A": "C", "C": "A", "B": "D", "D": "B"}
        new_position = target_quadrant["position"].replace("top", "TEMP").replace("bottom", "top").replace("TEMP", "bottom")
        problem = f"After flipping this image vertically, what letter will be in the {new_position} quadrant? Put your final answer in <boxed></boxed> tags."
        answer = f"<boxed>{target_quadrant['name']}</boxed>"

    expected_tool_calls = [
        {
            "tool": "image_flip_tool",
            "parameters": {
                "direction": direction
            }
        }
    ]

    metadata = {
        "quadrants": quadrants,
        "flip_direction": direction,
        "target_quadrant": target_quadrant,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "flip_quadrants"
    }

    return img, problem, answer, expected_tool_calls, metadata


def generate_number_sequence_scene():
    """Generate scene with numbers that become mirrored/inverted when flipped."""
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 80)
        small_font = ImageFont.truetype(FONT_PATH, 24)
    except Exception:
        font = small_font = ImageFont.load_default()

    # Draw a sequence of numbers horizontally
    numbers = [str(random.randint(1, 9)) for _ in range(5)]
    sequence = "".join(numbers)
    
    # Position sequence in center
    text_bbox = draw.textbbox((0, 0), sequence, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (IMG_WIDTH - text_width) // 2
    text_y = IMG_HEIGHT // 2 - 40
    
    draw.text((text_x, text_y), sequence, fill=(0, 0, 0), font=font)
    
    draw.text((IMG_WIDTH // 2 - 30, 30), "TOP", fill=(100, 100, 100), font=small_font)
    draw.text((IMG_WIDTH // 2 - 50, IMG_HEIGHT - 60), "BOTTOM", fill=(100, 100, 100), font=small_font)

    direction = random.choice(["horizontal", "vertical"])
    
    if direction == "horizontal":
        # Horizontal flip reverses the sequence
        reversed_sequence = sequence[::-1]
        problem = f"The image shows the number sequence '{sequence}'. After flipping horizontally, what will be the first digit you see from left to right? Put your final answer in <boxed></boxed> tags."
        answer = f"<boxed>{reversed_sequence[0]}</boxed>"
    else:
        # Vertical flip inverts but sequence order stays the same (reading from left)
        problem = f"The image shows the number sequence '{sequence}'. After flipping vertically, will the 'TOP' label appear at the top or bottom of the image? Put your final answer in <boxed></boxed> tags."
        answer = "<boxed>bottom</boxed>"

    expected_tool_calls = [
        {
            "tool": "image_flip_tool",
            "parameters": {
                "direction": direction
            }
        }
    ]

    metadata = {
        "sequence": sequence,
        "flip_direction": direction,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "flip_numbers"
    }

    return img, problem, answer, expected_tool_calls, metadata


def generate_asymmetric_shape_scene():
    """Generate scene with asymmetric shapes in specific positions."""
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    shapes = []
    
    # Circle in top-left
    circle_color = random.choice(COLORS)
    draw.ellipse([100, 100, 200, 200], fill=circle_color, outline=(0, 0, 0), width=2)
    shapes.append({"type": "circle", "color": rgb_to_color_name(circle_color), "position": "top-left"})
    
    # Square in bottom-right
    square_color = random.choice([c for c in COLORS if c != circle_color])
    draw.rectangle([600, 400, 700, 500], fill=square_color, outline=(0, 0, 0), width=2)
    shapes.append({"type": "square", "color": rgb_to_color_name(square_color), "position": "bottom-right"})
    
    # Triangle in top-right
    triangle_color = random.choice([c for c in COLORS if c not in [circle_color, square_color]])
    points = [(650, 100), (600, 200), (700, 200)]
    draw.polygon(points, fill=triangle_color, outline=(0, 0, 0), width=2)
    shapes.append({"type": "triangle", "color": rgb_to_color_name(triangle_color), "position": "top-right"})

    direction = random.choice(["horizontal", "vertical"])
    
    if direction == "horizontal":
        # After horizontal flip: circle goes to top-right, triangle goes to top-left, square goes to bottom-left
        problem = f"There's a {rgb_to_color_name(circle_color)} circle in the top-left. After flipping horizontally, which corner will the circle be in? Put your final answer in <boxed></boxed> tags."
        answer = "<boxed>top-right</boxed>"
    else:
        # After vertical flip: circle goes to bottom-left, square goes to top-right
        problem = f"There's a {rgb_to_color_name(square_color)} square in the bottom-right. After flipping vertically, which corner will the square be in? Put your final answer in <boxed></boxed> tags."
        answer = "<boxed>top-right</boxed>"

    expected_tool_calls = [
        {
            "tool": "image_flip_tool",
            "parameters": {
                "direction": direction
            }
        }
    ]

    metadata = {
        "shapes": shapes,
        "flip_direction": direction,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "flip_shapes"
    }

    return img, problem, answer, expected_tool_calls, metadata



def save_as_verl_parquet(rows, output_dir, split_name="train"):
    """
    Writes parquet with nested columns preserved (no json.dumps).
    """
    os.makedirs(output_dir, exist_ok=True)

    pa_rows = []
    for r in rows:
        pa_rows.append({
            "prompt": r["prompt"],
            "answer": r["answer"],
            "images": r["images"],
            "expected_tool_calls": r.get("expected_tool_calls", []),
            "metadata": r.get("metadata", {}),
            "data_source": r.get("data_source", "flip"),
            "reward_model": r.get("reward_model", {}),
            "agent_name": r.get("agent_name", "tool_agent"),
            "tools_kwargs": r.get("tools_kwargs", {}),
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
    parser = argparse.ArgumentParser(description="Generate flip dataset and save as VERL parquet")
    parser.add_argument("--output_dir", type=str, default="data/flip", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=200, help="Total samples to generate (train+test)")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction for train split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    generators = [
        generate_asymmetric_arrow_scene,
        generate_labeled_quadrants_scene,
        generate_number_sequence_scene,
        generate_asymmetric_shape_scene,
    ]

    print(f"Generating {args.num_samples} samples...")
    rows = []
    for i in tqdm(range(args.num_samples), desc="Generating"):
        gen = generators[i % len(generators)]
        img, problem, answer, expected_tool_calls, metadata = gen()

        # save image
        img_name = f"flip_{i:05d}.png"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        img_abs = os.path.abspath(img_path)

        tool_name = "image_flip_tool"

        user_content = (
            "<image>\n"
            f"{problem}\n\n"
            f"Use the {tool_name} to transform the image, then provide your answer."
        )
        
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tools_kwargs = {
            tool_name: {
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
            "data_source": "flip",
            "reward_model": reward_model,
            "agent_name": "tool_agent",
            "tools_kwargs": tools_kwargs,
        }
        rows.append(row)

    # split
    train_n = int(args.train_frac * len(rows))
    train_rows = rows[:train_n]
    test_rows = rows[train_n:]

    print(f"Saving {len(train_rows)} train and {len(test_rows)} test examples to parquet...")
    train_path, train_count = save_as_verl_parquet(train_rows, args.output_dir, split_name="train")
    test_path, test_count = save_as_verl_parquet(test_rows, args.output_dir, split_name="test")

    print("\n✓ Written parquet files:")
    print(f"  Train ({train_count}): {train_path}")
    print(f"  Test  ({test_count}): {test_path}")
    print(f"  Images dir: {images_dir}")

    # Verification
    print("\nRunning quick verification...")
    try:
        df = pd.read_parquet(train_path, engine="pyarrow")
        print("Columns:", df.columns.tolist())
        sample = df.iloc[0]
        print("Type reward_model (python):", type(sample["reward_model"]))
        print("Type expected_tool_calls (python):", type(sample["expected_tool_calls"]))

        counts = Counter([sample_meta.get("task_type", "unknown") for sample_meta in df["metadata"]])
        print("Task distribution (train):")
        for t, c in counts.items():
            print(f"  {t}: {c}")
    except Exception as e:
        print("Verification failed:", e)

    print("\nDataset ready. Point VERL config to these files:")
    print(f"  train_files: {train_path}")
    print(f"  val_files:   {test_path}")


if __name__ == "__main__":
    main()