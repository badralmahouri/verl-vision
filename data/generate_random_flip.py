#!/usr/bin/env python3
"""
Generate flip images + VERL-compatible parquet dataset for image_flip_tool training.

Creates tasks where the model must:
1. Analyze the image to identify that text is transformed
2. Use the flip tool to correct the transformation
3. Read the word after correction

No hints are provided about flip direction - the model must figure it out.
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

# Words to use - simple, common words
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

When analyzing an image:
1. Carefully examine what you see
2. If the content is hard to read, use the image_flip_tool to adjust the orientation
3. Provide your final answer in <boxed></boxed> tags

Think step by step about what transformation might help."""


def get_font(size):
    for font_path in [FONT_PATH] :
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()



def generate_horizontal_flip_scene():
    """Generate a word flipped horizontally (mirrored left-right)."""
    word = random.choice(WORDS)
    bg_color = random.choice(BACKGROUND_COLORS)
    text_color = random.choice(TEXT_COLORS)
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    font_size = 120
    font = get_font(font_size)
    
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
    
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # No hint labels - model must analyze visually
    
    problem = "What word is shown in this image?  Put your final answer in <boxed></boxed> tags."
    answer = f"<boxed>{word}</boxed>"
    
    expected_tool_calls = [
        {"tool": "image_flip_tool", "parameters": {"direction": "horizontal"}}
    ]
    
    metadata = {
        "word": word,
        "flip_direction": "horizontal",
        "task_type": "read_flipped_word"
    }
    
    return img, problem, answer, expected_tool_calls, metadata


def generate_vertical_flip_scene():
    """Generate a word flipped vertically (upside down)."""
    word = random.choice(WORDS)
    bg_color = random.choice(BACKGROUND_COLORS)
    text_color = random.choice(TEXT_COLORS)
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    font_size = 120
    font = get_font(font_size)
    
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
    
    # Flip the image vertically
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    
    problem = "What word is shown in this image? Put your final answer in <boxed></boxed> tags."
    answer = f"<boxed>{word}</boxed>"
    
    expected_tool_calls = [
        {"tool": "image_flip_tool", "parameters": {"direction": "vertical"}}
    ]
    
    metadata = {
        "word": word,
        "flip_direction": "vertical",
        "task_type": "read_flipped_word"
    }
    
    return img, problem, answer, expected_tool_calls, metadata


def generate_both_flips_scene():
    """Generate a word flipped both horizontally and vertically (rotated 180°)."""
    word = random.choice(WORDS)
    bg_color = random.choice(BACKGROUND_COLORS)
    text_color = random.choice(TEXT_COLORS)
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    font_size = 120
    font = get_font(font_size)
    
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
    
    # Flip both ways (equivalent to 180° rotation)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    
    first_flip = random.choice(["horizontal", "vertical"])
    second_flip = "vertical" if first_flip == "horizontal" else "horizontal"
    
    problem = "What word is shown in this image? Put your final answer in <boxed></boxed> tags."
    answer = f"<boxed>{word}</boxed>"
    
    expected_tool_calls = [
        {"tool": "image_flip_tool", "parameters": {"direction": first_flip}},
        {"tool": "image_flip_tool", "parameters": {"direction": second_flip}}
    ]
    
    metadata = {
        "word": word,
        "flip_direction": "both",
        "first_flip": first_flip,
        "second_flip": second_flip,
        "task_type": "read_flipped_word"
    }
    
    return img, problem, answer, expected_tool_calls, metadata


def generate_multiple_words_scene():
    """Generate multiple words, one flipped differently."""
    words = random.sample(WORDS, 3)
    bg_color = random.choice(BACKGROUND_COLORS)
    text_color = random.choice(TEXT_COLORS)
    
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)
    
    font_size = 60
    font = get_font(font_size)
    
    y_positions = [80, 170, 260]
    flipped_index = random.randint(0, 2)
    flip_direction = random.choice(["horizontal", "vertical"])
    
    for i, (word, y) in enumerate(zip(words, y_positions)):
        bbox = draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        x = (IMG_WIDTH - text_width) // 2
        
        if i == flipped_index:
            # Create a small image for this word and flip it
            word_img = Image.new('RGB', (text_width + 20, 80), bg_color)
            word_draw = ImageDraw.Draw(word_img)
            word_draw.text((10, 10), word, fill=text_color, font=font)
            
            if flip_direction == "horizontal":
                word_img = word_img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                word_img = word_img.transpose(Image.FLIP_TOP_BOTTOM)
            
            img.paste(word_img, (x - 10, y))
        else:
            draw.text((x, y + 10), word, fill=text_color, font=font)
    
    
    problem = "This image shows three words. One of them appears different. Tell me what all the words say. Put your final answer in <boxed></boxed> tags."
    answer = f"<boxed>{words[flipped_index]}</boxed>"
    
    expected_tool_calls = [
        {"tool": "image_flip_tool", "parameters": {"direction": flip_direction}}
    ]
    
    metadata = {
        "words": words,
        "flipped_word": words[flipped_index],
        "flipped_index": flipped_index,
        "flip_direction": flip_direction,
        "task_type": "find_flipped_word"
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
    parser = argparse.ArgumentParser(description="Generate flip word dataset for VERL")
    parser.add_argument("--output_dir", type=str, default="data/flip", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=200, help="Total samples to generate")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction for train split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    generators = [
        (generate_horizontal_flip_scene, 3),    # 3x weight
        (generate_vertical_flip_scene, 3),       # 3x weight
        (generate_both_flips_scene, 2),          # 2x weight
        (generate_multiple_words_scene, 2),      # 2x weight
    ]
    
    weighted_generators = []
    for gen, weight in generators:
        weighted_generators.extend([gen] * weight)

    print(f"Generating {args.num_samples} samples...")
    rows = []
    for i in tqdm(range(args.num_samples), desc="Generating"):
        gen = random.choice(weighted_generators)
        img, problem, answer, expected_tool_calls, metadata = gen()

        img_name = f"flip_{i:05d}.png"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        img_abs = os.path.abspath(img_path)

        tool_name = "image_flip_tool"

        user_content = f"<image>\n{problem}"
        
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

    random.shuffle(rows)
    
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
        df = pd.read_parquet(train_path)
        print(f"Columns: {df.columns.tolist()}")
        
        counts = Counter([m.get("task_type", "unknown") for m in df["metadata"]])
        flip_counts = Counter([m.get("flip_direction", "unknown") for m in df["metadata"]])
        
        print("\nTask distribution:")
        for t, c in counts.items():
            print(f"  {t}: {c}")
        
        print("\nFlip direction distribution:")
        for t, c in flip_counts.items():
            print(f"  {t}: {c}")
            
        sample = df.iloc[0]
        print(f"\nSample word: {sample['metadata'].get('word', 'N/A')}")
        print(f"Sample answer: {sample['answer']}")
        
    except Exception as e:
        print(f"Verification failed: {e}")

    print(f"  data.train_files={train_path}")
    print(f"  data.val_files={test_path}")


if __name__ == "__main__":
    main()
