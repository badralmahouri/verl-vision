#!/usr/bin/env python3
"""
Generate flip + rotate images + VERL-compatible parquet dataset.
Creates tasks where the model must use flip and/or rotate tools to read words.
Extends the flip_geometry style to include rotation tasks.
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
IMG_HEIGHT = 400

BACKGROUND_COLORS = [
    (255, 255, 255),  # white
    (240, 240, 240),  # light gray
    (255, 250, 240),  # floral white
    (240, 255, 240),  # honeydew
    (240, 248, 255),  # alice blue
]

TEXT_COLORS = [
    (0, 0, 0),        # black
    (50, 50, 50),     # dark gray
    (0, 0, 139),      # dark blue
    (139, 0, 0),      # dark red
    (0, 100, 0),      # dark green
]

WORDS = [
    # Short words
    "CAT", "DOG", "SUN", "MOON", "STAR", "BIRD", "FISH", "TREE",
    "BOOK", "CAKE", "DOOR", "FIRE", "GOLD", "HAND", "KING", "LAMP",
    "MILK", "NEST", "RAIN", "ROAD", "SHIP", "TIME", "WAVE", "WIND",
    # Medium words
    "APPLE", "BEACH", "CHAIR", "DANCE", "EARTH", "FLAME", "GRASS",
    "HOUSE", "JUICE", "KNIFE", "LEMON", "MAGIC", "NIGHT", "OCEAN",
    "PEACE", "QUEEN", "RIVER", "STORM", "TOWER", "WATER", "ZEBRA",
    # Longer words
    "BUTTERFLY", "CHOCOLATE", "ELEPHANT", "MOUNTAIN", "SUNSHINE",
    "RAINBOW", "DIAMOND", "FLOWER", "GARDEN", "PLANET", "SUMMER",
]

SYSTEM_PROMPT = """You are a helpful assistant that analyzes images using tools.

When you see a flipped, mirrored, or rotated word in an image:
1. Use the image_flip_tool to flip the image (horizontal or vertical)
2. Use the image_rotate_tool to rotate the image by an angle 
3. After transforming the image, read the word and provide your final answer in <boxed></boxed> tags

Always use the appropriate tool(s) before answering."""


def get_font(size):
    """Try to load a font, with fallbacks."""
    for font_path in [FONT_PATH]:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_word_image(word, bg_color=None, text_color=None, font_size=120):
    """Create a base image with a word centered."""
    if bg_color is None:
        bg_color = random.choice(BACKGROUND_COLORS)
    if text_color is None:
        text_color = random.choice(TEXT_COLORS)
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    font = get_font(font_size)
    
    # Get text size and adjust if needed
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    while text_width > IMG_WIDTH - 100 and font_size > 40:
        font_size -= 10
        font = get_font(font_size)
        bbox = draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    
    x = (IMG_WIDTH - text_width) // 2
    y = (IMG_HEIGHT - text_height) // 2
    
    draw.text((x, y), word, fill=text_color, font=font)
    
    return img


def add_hint_label(img, hint_text):
    """Add a small hint label to the image."""
    draw = ImageDraw.Draw(img)
    small_font = get_font(20)
    draw.text((10, 10), hint_text, fill=(150, 150, 150), font=small_font)
    return img




def generate_horizontal_flip_scene():
    """Generate a word flipped horizontally (mirrored left-right)."""
    word = random.choice(WORDS)
    img = create_word_image(word)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    problem = "This image shows a transformed word. Use the available tools to restore the image and read the word. What word is shown? Put your final answer in <boxed></boxed> tags."
    answer = word
    
    expected_tool_calls = [
        {"tool": "image_flip_tool", "parameters": {"direction": "horizontal"}}
    ]
    
    metadata = {
        "word": word,
        "transform_type": "flip_horizontal",
        "task_type": "read_transformed_word"
    }
    
    return img, problem, answer, expected_tool_calls, metadata


def generate_vertical_flip_scene():
    """Generate a word flipped vertically (upside down)."""
    word = random.choice(WORDS)
    img = create_word_image(word)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    problem = "This image shows a transformed word. Use the available tools to restore the image and read the word. What word is shown? Put your final answer in <boxed></boxed> tags."
    answer = word
    
    expected_tool_calls = [
        {"tool": "image_flip_tool", "parameters": {"direction": "vertical"}}
    ]
    
    metadata = {
        "word": word,
        "transform_type": "flip_vertical",
        "task_type": "read_transformed_word"
    }
    
    return img, problem, answer, expected_tool_calls, metadata



def generate_rotate_scene():
    """Generate a word rotated by a random angle."""
    word = random.choice(WORDS)
    img = create_word_image(word)
    
    angle = random.randint(0, 360)

    
    img = img.rotate(-angle, expand=True, fillcolor=random.choice(BACKGROUND_COLORS))
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    
    undo_angle = (360 - angle) % 360
    if undo_angle == 0:
        undo_angle = 360
    
    problem = "This image shows a rotated word. What word is shown? Put your final answer in <boxed></boxed> tags."
    answer = word
    
    expected_tool_calls = [
        {"tool": "image_rotate_tool", "parameters": {"angle": undo_angle}}
    ]
    
    metadata = {
        "word": word,
        "transform_type": f"rotate_{angle}",
        "rotate_angle": angle,
        "undo_angle": undo_angle,
        "task_type": "read_transformed_word"
    }
    
    return img, problem, answer, expected_tool_calls, metadata


def generate_flip_then_rotate_scene():
    """Generate a word that is flipped then rotated."""
    word = random.choice(WORDS)
    img = create_word_image(word)
    
    flip_dir = random.choice(["horizontal", "vertical"])
    rotate_angle = random.choice([45, 90, 135, 180, 225, 270, 315])
    
    if flip_dir == "horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    if rotate_angle in [90, 270]:
        img = img.rotate(-rotate_angle, expand=True)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    else:
        img = img.rotate(-rotate_angle, expand=True, fillcolor=random.choice(BACKGROUND_COLORS))
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    
    
    undo_rotate = (360 - rotate_angle) % 360
    if undo_rotate == 0:
        undo_rotate = 360
    
    problem = "This image shows a word that has been transformed (possibly flipped and/or rotated). Use the available tools to restore the image and read the word. What word is shown? Put your final answer in <boxed></boxed> tags."
    answer = word
    
    expected_tool_calls = [
        {"tool": "image_rotate_tool", "parameters": {"angle": undo_rotate if rotate_angle != 180 else 180}},
        {"tool": "image_flip_tool", "parameters": {"direction": flip_dir}}
    ]
    
    metadata = {
        "word": word,
        "transform_type": "flip_then_rotate",
        "flip_direction": flip_dir,
        "rotate_angle": rotate_angle,
        "task_type": "read_transformed_word"
    }
    
    return img, problem, answer, expected_tool_calls, metadata


def generate_rotate_then_flip_scene():
    """Generate a word that is rotated then flipped."""
    word = random.choice(WORDS)
    img = create_word_image(word)
    
    rotate_angle = random.choice([45, 90, 135, 180, 225, 270, 315])
    flip_dir = random.choice(["horizontal", "vertical"])
    
    img = img.rotate(-rotate_angle, expand=True, fillcolor=random.choice(BACKGROUND_COLORS))
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    
    if flip_dir == "horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    
    # Undo: first flip back, then rotate back
    undo_rotate = (360 - rotate_angle) % 360
    if undo_rotate == 0:
        undo_rotate = 360
    
    problem = "This image shows a word that has been transformed (possibly flipped and/or rotated). Use the available tools to restore the image and read the word. What word is shown? Put your final answer in <boxed></boxed> tags."
    answer = word
    
    expected_tool_calls = [
        {"tool": "image_flip_tool", "parameters": {"direction": flip_dir}},
        {"tool": "image_rotate_tool", "parameters": {"angle": undo_rotate}}
    ]
    
    metadata = {
        "word": word,
        "transform_type": "rotate_then_flip",
        "rotate_angle": rotate_angle,
        "flip_direction": flip_dir,
        "task_type": "read_transformed_word"
    }
    
    return img, problem, answer, expected_tool_calls, metadata



def save_as_verl_parquet(rows, output_dir, split_name="train"):
    """Writes parquet with nested columns preserved."""
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
    parser = argparse.ArgumentParser(description="Generate flip+rotate word dataset for VERL")
    parser.add_argument("--output_dir", type=str, default="data/flip_rotate", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=300, help="Total samples to generate")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction for train split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    generators = [
        (generate_horizontal_flip_scene, 2),
        (generate_vertical_flip_scene, 2),
        (generate_rotate_scene, 6),  
        (generate_flip_then_rotate_scene, 4),
        (generate_rotate_then_flip_scene, 4),
    ]
    
    weighted_generators = []
    for gen, weight in generators:
        weighted_generators.extend([gen] * weight)

    print(f"Generating {args.num_samples} samples...")
    rows = []
    for i in tqdm(range(args.num_samples), desc="Generating"):
        gen = random.choice(weighted_generators)
        img, problem, answer, expected_tool_calls, metadata = gen()

        img_name = f"transform_{i:05d}.png"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        img_abs = os.path.abspath(img_path)

        user_content = f"<image>\n{problem}"
        
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tools_kwargs = {
            "image_flip_tool": {
                "create_kwargs": {"image": img_abs}
            },
            "image_rotate_tool": {
                "create_kwargs": {"image": img_abs}
            }
        }

        reward_model = {
            "style": "rule",
            "ground_truth": f"<boxed>{answer}</boxed>",
            "expected_tool_calls": expected_tool_calls
        }

        row = {
            "prompt": prompt,
            "answer": f"<boxed>{answer}</boxed>",
            "images": [img_abs],
            "expected_tool_calls": expected_tool_calls,
            "metadata": {**metadata, "tools_kwargs": tools_kwargs},
            "data_source": "flip",  # Reuse flip reward function
            "reward_model": reward_model,
            "agent_name": "tool_agent",
            "tools_kwargs": tools_kwargs,
        }
        rows.append(row)

    random.shuffle(rows)
    
    train_n = int(args.train_frac * len(rows))
    train_rows = rows[:train_n]
    test_rows = rows[train_n:]

    print(f"Saving {len(train_rows)} train and {len(test_rows)} test examples...")
    train_path, train_count = save_as_verl_parquet(train_rows, args.output_dir, split_name="train")
    test_path, test_count = save_as_verl_parquet(test_rows, args.output_dir, split_name="test")

    print("\n Dataset created:")
    print(f"  Train ({train_count}): {train_path}")
    print(f"  Test  ({test_count}): {test_path}")
    print(f"  Images: {images_dir}")

    try:
        df = pd.read_parquet(train_path)
        print(f"Columns: {df.columns.tolist()}")
        
        transform_counts = Counter([m.get("transform_type", "unknown") for m in df["metadata"]])
        
        print("\nTransform type distribution:")
        for t, c in sorted(transform_counts.items()):
            print(f"  {t}: {c}")
            
        # Show sample
        sample = df.iloc[0]
        print(f"\nSample word: {sample['metadata'].get('word', 'N/A')}")
        print(f"Sample transform: {sample['metadata'].get('transform_type', 'N/A')}")
        print(f"Sample answer: {sample['answer']}")
        
    except Exception as e:
        print(f"Verification failed: {e}")
    print(f"  data.train_files={train_path}")
    print(f"  data.val_files={test_path}")


if __name__ == "__main__":
    main()
