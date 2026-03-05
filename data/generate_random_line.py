#!/usr/bin/env python3
"""
Generate line drawing dataset - REWARD-FOCUSED DESIGN.
The reward model verifies the LINE TOOL was used correctly, not answer accuracy.
"""

import os
import random
import argparse
import math
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

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "orange": (255, 128, 0),
    "purple": (128, 0, 255),
}

SYSTEM_PROMPT = """You are an assistant that draws lines on images using the image_line_tool.

When asked to draw a line:
1. Identify the start and end coordinates from the image
2. Call image_line_tool with the correct coordinates
3. Confirm the line was drawn in <boxed></boxed> tags with "success"

Always use the tool to draw the requested line."""


def generate_labeled_points_scene():
    """
    Task: Connect two explicitly labeled points.
    Reward: Tool called with correct start/end coordinates (±tolerance).
    Success metric: Coordinate accuracy, not VLM reasoning.
    """
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
        coord_font = ImageFont.truetype(FONT_PATH, 14)
    except Exception:
        font = ImageFont.load_default()
        coord_font = ImageFont.load_default()

    num_points = random.randint(2, 4)
    points = []
    margin = 100
    
    for i in range(num_points):
        x = random.randint(margin, IMG_WIDTH - margin)
        y = random.randint(margin, IMG_HEIGHT - margin)
        
        point_size = 10
        color = random.choice(list(COLORS.values()))
        draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], 
                     fill=color, outline=(0, 0, 0), width=2)
        
        label = f"P{i + 1}"
        draw.text((x + 15, y - 25), label, fill=(0, 0, 0), font=font)
        draw.text((x + 15, y - 5), f"({x},{y})", fill=(100, 100, 100), font=coord_font)
        
        points.append({
            "label": label,
            "coords": [x, y]
        })

    p1, p2 = random.sample(points, 2)
    
    problem = f"Draw a line from {p1['label']} to {p2['label']}. Confirm with <boxed>success</boxed> when done."
    answer = "<boxed>success</boxed>"

    expected_tool_calls = [{
        "tool": "image_line_tool",
        "parameters": {
            "start": p1["coords"],
            "end": p2["coords"]
        }
    }]

    metadata = {
        "task_type": "labeled_points",
        "all_points": points,
        "target_start": p1,
        "target_end": p2,
        "coordinate_tolerance": 15  # pixels tolerance for reward
    }

    return img, problem, answer, expected_tool_calls, metadata


def generate_color_to_color_scene():
    """
    Task: Connect two colored regions.
    Reward: Tool called with coordinates inside correct colored regions.
    Forces VLM to do visual grounding + tool use.
    """
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except Exception:
        font = ImageFont.load_default()

    num_circles = random.randint(3, 4)
    circles = []
    margin = 120
    
    color_names = list(COLORS.keys())
    selected_colors = random.sample(color_names, num_circles)
    
    positions = [
        (150, 150), (650, 150), (150, 450), (650, 450), (400, 300)
    ]
    
    for i, (pos, color_name) in enumerate(zip(random.sample(positions, num_circles), selected_colors)):
        x, y = pos
        radius = random.randint(40, 60)
        color = COLORS[color_name]
        
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline=(0, 0, 0), width=3)
        
        circles.append({
            "color": color_name,
            "center": [x, y],
            "radius": radius
        })

    c1, c2 = random.sample(circles, 2)
    
    problem = f"Draw a line from the {c1['color']} circle to the {c2['color']} circle. Confirm with <boxed>success</boxed> when done."
    answer = "<boxed>success</boxed>"

    expected_tool_calls = [{
        "tool": "image_line_tool",
        "parameters": {
            "start": c1["center"],
            "end": c2["center"]
        }
    }]

    metadata = {
        "task_type": "color_to_color",
        "all_circles": circles,
        "target_start": c1,
        "target_end": c2,
        "coordinate_tolerance": 50  # larger tolerance since no exact coords given
    }

    return img, problem, answer, expected_tool_calls, metadata


def generate_geometric_instruction_scene():
    """
    Task: Follow geometric instruction (e.g., "draw horizontal line across center").
    Reward: Tool called with coordinates matching the geometric constraint.
    Tests spatial reasoning + tool execution.
    """
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, 22)
    except Exception:
        font = ImageFont.load_default()

    # Draw reference grid
    grid_color = (200, 200, 200)
    step = 100
    for x in range(0, IMG_WIDTH, step):
        draw.line([(x, 0), (x, IMG_HEIGHT)], fill=grid_color, width=1)
    for y in range(0, IMG_HEIGHT, step):
        draw.line([(0, y), (IMG_WIDTH, y)], fill=grid_color, width=1)
    
    # Add center markers
    center_x, center_y = IMG_WIDTH // 2, IMG_HEIGHT // 2
    draw.line([(center_x - 20, center_y), (center_x + 20, center_y)], fill=(255, 0, 0), width=2)
    draw.line([(center_x, center_y - 20), (center_x, center_y + 20)], fill=(255, 0, 0), width=2)
    draw.text((center_x + 25, center_y - 10), "CENTER", fill=(255, 0, 0), font=font)

    # Choose instruction type
    instruction_type = random.choice(["horizontal", "vertical", "diagonal"])
    
    if instruction_type == "horizontal":
        y = center_y
        start = [100, y]
        end = [IMG_WIDTH - 100, y]
        problem = f"Draw a horizontal line through the center of the image. Confirm with <boxed>success</boxed> when done."
        tolerance = 30  # y-coordinate tolerance
        
    elif instruction_type == "vertical":
        x = center_x
        start = [x, 50]
        end = [x, IMG_HEIGHT - 50]
        problem = f"Draw a vertical line through the center of the image. Confirm with <boxed>success</boxed> when done."
        tolerance = 30  # x-coordinate tolerance
        
    else:  # diagonal
        start = [100, 100]
        end = [IMG_WIDTH - 100, IMG_HEIGHT - 100]
        problem = f"Draw a diagonal line from the top-left quadrant to the bottom-right quadrant. Confirm with <boxed>success</boxed> when done."
        tolerance = 50

    answer = "<boxed>success</boxed>"

    expected_tool_calls = [{
        "tool": "image_line_tool",
        "parameters": {
            "start": start,
            "end": end
        }
    }]

    metadata = {
        "task_type": "geometric_instruction",
        "instruction": instruction_type,
        "expected_start": start,
        "expected_end": end,
        "coordinate_tolerance": tolerance
    }

    return img, problem, answer, expected_tool_calls, metadata


def generate_visual_grounding_scene():
    """
    Task: Connect specific visual elements (shapes, objects).
    Reward: Tool called with coordinates near the specified objects.
    Pure visual grounding test.
    """
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, 18)
    except Exception:
        font = ImageFont.load_default()

    # Create different shapes
    shapes = []
    shape_configs = [
        ("circle", (200, 200), 50),
        ("square", (600, 200), 60),
        ("triangle", (200, 450), 70),
        ("star", (600, 450), 55),
    ]
    
    selected = random.sample(shape_configs, 3)
    
    for shape_type, (cx, cy), size in selected:
        color = random.choice(list(COLORS.values()))
        
        if shape_type == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(0, 0, 0), width=2)
            center = [cx, cy]
            
        elif shape_type == "square":
            draw.rectangle([cx - size, cy - size, cx + size, cy + size],
                          fill=color, outline=(0, 0, 0), width=2)
            center = [cx, cy]
            
        elif shape_type == "triangle":
            points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
            center = [cx, cy]
            
        else: 
            points = [
                (cx, cy - size),
                (cx + size, cy - size//3),
                (cx + size//2, cy + size),
                (cx - size//2, cy + size),
                (cx - size, cy - size//3)
            ]
            draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
            center = [cx, cy]
        
        shapes.append({
            "type": shape_type,
            "center": center,
            "size": size
        })

    s1, s2 = random.sample(shapes, 2)
    
    problem = f"Draw a line connecting the {s1['type']} to the {s2['type']}. Confirm with <boxed>success</boxed> when done."
    answer = "<boxed>success</boxed>"

    expected_tool_calls = [{
        "tool": "image_line_tool",
        "parameters": {
            "start": s1["center"],
            "end": s2["center"]
        }
    }]

    metadata = {
        "task_type": "visual_grounding",
        "all_shapes": shapes,
        "target_start": s1,
        "target_end": s2,
        "coordinate_tolerance": max(s1["size"], s2["size"]) // 2  
    }

    return img, problem, answer, expected_tool_calls, metadata



def save_as_verl_parquet(rows, output_dir, split_name="train"):
    """Save dataset with proper VERL structure."""
    os.makedirs(output_dir, exist_ok=True)

    pa_rows = []
    for r in rows:
        # CRITICAL: extra_info must contain ALL data needed by reward function

        extra_info = {
            "expected_tool_calls": r.get("expected_tool_calls", []),
            "metadata": r.get("metadata", {}),
            "tools_kwargs": r.get("tools_kwargs", {}), 
        }
        
        pa_rows.append({
            "prompt": r["prompt"],
            "answer": r["answer"],
            "images": r["images"],
            "expected_tool_calls": r.get("expected_tool_calls", []),
            "metadata": r.get("metadata", {}),
            "data_source": r.get("data_source", "line"),
            "reward_model": r.get("reward_model", {}),
            "agent_name": r.get("agent_name", "tool_agent"),
            "tools_kwargs": r.get("tools_kwargs", {}),
            "extra_info": extra_info,
        })

    table = pa.Table.from_pylist(pa_rows)
    out_path = os.path.join(output_dir, f"{split_name}.parquet")
    pq.write_table(table, out_path)
    return out_path, len(pa_rows)



def main():
    parser = argparse.ArgumentParser(description="Generate line drawing dataset (reward-focused)")
    parser.add_argument("--output_dir", type=str, default="data/line", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=200, help="Total samples")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    generators = [
        generate_labeled_points_scene,
        generate_color_to_color_scene,
        generate_geometric_instruction_scene,
        generate_visual_grounding_scene,
    ]

    print(f"Generating {args.num_samples} samples (reward-focused design)...")
    rows = []
    
    for i in tqdm(range(args.num_samples), desc="Generating"):
        gen = generators[i % len(generators)]
        img, problem, answer, expected_tool_calls, metadata = gen()

        img_name = f"line_{i:05d}.png"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        img_abs = os.path.abspath(img_path)

        user_content = f"<image>\n{problem}"
        
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tools_kwargs = {
            "image_line_tool": {
                "create_kwargs": {
                    "image": img_abs,
                    "expected_tool_calls": expected_tool_calls,
                    "coordinate_tolerance": metadata.get("coordinate_tolerance", 50),
                }
            }
        }

        reward_model = {
            "style": "tool_verification", 
            "ground_truth": answer,  
            "expected_tool_calls": expected_tool_calls,  
            "verification": {
                "check_tool_called": True,  
                "check_coordinates": True,  
                "coordinate_tolerance": metadata.get("coordinate_tolerance", 20),
                "check_answer": True,  
            }
        }

        row = {
            "prompt": prompt,
            "answer": answer,
            "images": [img_abs],
            "expected_tool_calls": expected_tool_calls,
            "metadata": {**metadata, "tools_kwargs": tools_kwargs},
            "data_source": "line",
            "reward_model": reward_model,
            "agent_name": "tool_agent",
            "tools_kwargs": tools_kwargs,
        }
        rows.append(row)

    train_n = int(args.train_frac * len(rows))
    train_rows = rows[:train_n]
    test_rows = rows[train_n:]

    print(f"Saving {len(train_rows)} train and {len(test_rows)} test examples...")
    train_path, train_count = save_as_verl_parquet(train_rows, args.output_dir, split_name="train")
    test_path, test_count = save_as_verl_parquet(test_rows, args.output_dir, split_name="test")

    print(f"  Train ({train_count}): {train_path}")
    print(f"  Test  ({test_count}): {test_path}")
    print(f"  Images: {images_dir}")

    try:
        df = pd.read_parquet(train_path, engine="pyarrow")
        counts = Counter([m.get("task_type", "unknown") for m in df["metadata"]])
        print("Task distribution:")
        for t, c in counts.items():
            print(f"  {t}: {c}")
        
        print(f"\nReward model style: {df.iloc[0]['reward_model']['style']}")
        print(f"Verification checks: {df.iloc[0]['reward_model']['verification']}")
    except Exception as e:
        print(f"Verification error: {e}")



if __name__ == "__main__":
    main()
