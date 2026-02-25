#!/usr/bin/env python3
"""
Generate zoom/crop dataset for VERL training.
Shapes have random labels. Questions identify by label, ask about color or shape.
The answer is NEVER in the question.
"""

import os
import random
import argparse
import math
import string
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from collections import Counter

FONT_PATH = "arial.ttf"

def load_words():
    try:
        with open("words.txt", "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip() and len(line.strip()) > 3]
            return words[:500] 
    except:
        return ["apple", "banana", "cherry", "dog", "elephant", "flower", "guitar", "house", "island", "jungle"]

WORDS = load_words()

SYSTEM_PROMPT = """You are a helpful assistant that analyzes images using the zoom/crop tool.

When asked about specific regions or objects in an image:
1. Use image_zoom_in_tool to crop and inspect the target region closely
2. After receiving the tool response, provide your final answer in <boxed></boxed> tags

Always use the zoom tool before answering questions about image regions."""

# Shape types
SHAPES = ['star', 'heart', 'arrow', 'circle', 'square', 'diamond', 'cross', 'ring', 'triangle', 'pentagon']

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128),
]
COLOR_NAMES = {
    (255, 0, 0): "red", (0, 255, 0): "green", (0, 0, 255): "blue",
    (255, 255, 0): "yellow", (255, 0, 255): "magenta", (0, 255, 255): "cyan",
    (255, 128, 0): "orange", (128, 0, 255): "purple", (0, 255, 128): "lime",
    (128, 128, 0): "olive", (0, 128, 128): "teal", (128, 0, 128): "violet",
}


def random_color():
    return random.choice(COLORS)


def rgb_to_name(rgb):
    return COLOR_NAMES.get(tuple(rgb), "mixed")


def generate_label():
    """Generate a random label like A1, B2, X7, etc."""
    letter = random.choice(string.ascii_uppercase)
    number = random.randint(1, 9)
    return f"{letter}{number}"


def generate_gradient_bg(w, h):
    img = Image.new('RGB', (w, h))
    c1 = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
    c2 = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
    for y in range(h):
        r = int(c1[0] + (c2[0] - c1[0]) * y / h)
        g = int(c1[1] + (c2[1] - c1[1]) * y / h)
        b = int(c1[2] + (c2[2] - c1[2]) * y / h)
        for x in range(w):
            img.putpixel((x, y), (r, g, b))
    return img


def generate_noise_bg(w, h):
    img = Image.new('RGB', (w, h))
    base = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
    for y in range(h):
        for x in range(w):
            noise = random.randint(-20, 20)
            img.putpixel((x, y), tuple(max(0, min(255, c + noise)) for c in base))
    return img


def generate_pattern_bg(w, h):
    base = (random.randint(200, 245), random.randint(200, 245), random.randint(200, 245))
    img = Image.new('RGB', (w, h), base)
    draw = ImageDraw.Draw(img)
    pattern = random.choice(['stripes', 'dots', 'grid', 'checkers'])
    color2 = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
    
    if pattern == 'stripes':
        step = random.randint(10, 40)
        for i in range(0, max(w, h) * 2, step * 2):
            if random.random() > 0.5:
                draw.line([(i, 0), (0, i)], fill=color2, width=random.randint(2, 8))
            else:
                draw.line([(0, i), (i, 0)], fill=color2, width=random.randint(2, 8))
    elif pattern == 'dots':
        step = random.randint(15, 40)
        r = random.randint(3, 8)
        for x in range(0, w, step):
            for y in range(0, h, step):
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color2)
    elif pattern == 'grid':
        step = random.randint(20, 60)
        for x in range(0, w, step):
            draw.line([(x, 0), (x, h)], fill=color2, width=1)
        for y in range(0, h, step):
            draw.line([(0, y), (w, y)], fill=color2, width=1)
    else:
        step = random.randint(20, 50)
        for i in range(0, w, step):
            for j in range(0, h, step):
                if (i // step + j // step) % 2 == 0:
                    draw.rectangle([i, j, i+step, j+step], fill=color2)
    return img


def generate_background(w, h):
    bg_type = random.choice(['solid', 'gradient', 'noise', 'pattern'])
    if bg_type == 'solid':
        return Image.new('RGB', (w, h), (random.randint(200, 245), random.randint(200, 245), random.randint(200, 245)))
    elif bg_type == 'gradient':
        return generate_gradient_bg(w, h)
    elif bg_type == 'noise':
        return generate_noise_bg(w, h)
    else:
        return generate_pattern_bg(w, h)


def draw_icon(size, color=None, shape_type=None):
    """Draw a shape icon with optional specified color and type."""
    size = max(40, size)
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    icon_type = shape_type if shape_type else random.choice(SHAPES)
    color = color if color else random_color()
    
    cx, cy = size // 2, size // 2
    r = size // 2 - 5
    
    if icon_type == 'star':
        points = []
        for i in range(5):
            angle = math.pi / 2 + i * 4 * math.pi / 5
            points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
            angle += 2 * math.pi / 5
            points.append((cx + r * 0.4 * math.cos(angle), cy - r * 0.4 * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'heart':
        draw.ellipse([cx - r, cy - r//2, cx, cy + r//2], fill=color, outline=(0, 0, 0))
        draw.ellipse([cx, cy - r//2, cx + r, cy + r//2], fill=color, outline=(0, 0, 0))
        draw.polygon([(cx - r + 5, cy), (cx + r - 5, cy), (cx, cy + r)], fill=color, outline=(0, 0, 0))
    elif icon_type == 'arrow':
        draw.polygon([(cx, cy - r), (cx + r//2, cy), (cx + r//4, cy), 
                      (cx + r//4, cy + r), (cx - r//4, cy + r), (cx - r//4, cy),
                      (cx - r//2, cy)], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'circle':
        draw.ellipse([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'square':
        draw.rectangle([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'diamond':
        draw.polygon([(cx, 5), (size-5, cy), (cx, size-5), (5, cy)], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'cross':
        w = r // 2
        draw.rectangle([cx - w, 5, cx + w, size - 5], fill=color, outline=(0, 0, 0), width=1)
        draw.rectangle([5, cy - w, size - 5, cy + w], fill=color, outline=(0, 0, 0), width=1)
    elif icon_type == 'ring':
        draw.ellipse([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0), width=2)
        inner = size // 3
        draw.ellipse([5 + inner, 5 + inner, size-5 - inner, size-5 - inner], fill=(255, 255, 255, 200), outline=(0, 0, 0), width=1)
    elif icon_type == 'triangle':
        draw.polygon([(cx, 5), (5, size-5), (size-5, size-5)], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'pentagon':
        points = [(cx + r * math.sin(2 * math.pi * k / 5), cy - r * math.cos(2 * math.pi * k / 5)) for k in range(5)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    else:
        draw.ellipse([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0), width=2)
    
    return img, icon_type, color


def draw_text_element(size, word=None, color=None):
    """Draw a text element with optional specified word and color."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    text = word if word else random.choice(WORDS)
    color = color if color else random_color()
    
    try:
        font = ImageFont.truetype(FONT_PATH, min(size // 2, 24))
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) // 2, (size - th) // 2), text, fill=color, font=font)
    
    return img, text, color


def draw_complex_shape(size):
    """Draw multiple overlapping shapes."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    num_shapes = random.randint(2, 5)
    colors = [random_color() for _ in range(num_shapes)]
    
    for i in range(num_shapes):
        x1 = random.randint(0, size // 2)
        y1 = random.randint(0, size // 2)
        x2 = random.randint(size // 2, size)
        y2 = random.randint(size // 2, size)
        
        if random.random() > 0.5:
            draw.ellipse([x1, y1, x2, y2], fill=colors[i], outline=(0, 0, 0))
        else:
            draw.rectangle([x1, y1, x2, y2], fill=colors[i], outline=(0, 0, 0))
    
    return img, "composite", colors[0]


def draw_labeled_icon(size, label, color=None, shape_type=None):
    """Draw a shape icon with a label on it."""
    size = max(50, size)
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    icon_type = shape_type if shape_type else random.choice(SHAPES)
    color = color if color else random_color()
    
    cx, cy = size // 2, size // 2
    r = size // 2 - 5
    
    # Draw the shape
    if icon_type == 'star':
        points = []
        for i in range(5):
            angle = math.pi / 2 + i * 4 * math.pi / 5
            points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
            angle += 2 * math.pi / 5
            points.append((cx + r * 0.4 * math.cos(angle), cy - r * 0.4 * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'heart':
        draw.ellipse([cx - r, cy - r//2, cx, cy + r//2], fill=color, outline=(0, 0, 0))
        draw.ellipse([cx, cy - r//2, cx + r, cy + r//2], fill=color, outline=(0, 0, 0))
        draw.polygon([(cx - r + 5, cy), (cx + r - 5, cy), (cx, cy + r)], fill=color, outline=(0, 0, 0))
    elif icon_type == 'arrow':
        draw.polygon([(cx, cy - r), (cx + r//2, cy), (cx + r//4, cy), 
                      (cx + r//4, cy + r), (cx - r//4, cy + r), (cx - r//4, cy),
                      (cx - r//2, cy)], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'circle':
        draw.ellipse([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'square':
        draw.rectangle([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'diamond':
        draw.polygon([(cx, 5), (size-5, cy), (cx, size-5), (5, cy)], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'cross':
        w = r // 2
        draw.rectangle([cx - w, 5, cx + w, size - 5], fill=color, outline=(0, 0, 0), width=1)
        draw.rectangle([5, cy - w, size - 5, cy + w], fill=color, outline=(0, 0, 0), width=1)
    elif icon_type == 'ring':
        draw.ellipse([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0), width=2)
        inner = size // 3
        draw.ellipse([5 + inner, 5 + inner, size-5 - inner, size-5 - inner], fill=(255, 255, 255, 200), outline=(0, 0, 0), width=1)
    elif icon_type == 'triangle':
        draw.polygon([(cx, 5), (5, size-5), (size-5, size-5)], fill=color, outline=(0, 0, 0), width=2)
    elif icon_type == 'pentagon':
        points = [(cx + r * math.sin(2 * math.pi * k / 5), cy - r * math.cos(2 * math.pi * k / 5)) for k in range(5)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    else:
        draw.ellipse([5, 5, size-5, size-5], fill=color, outline=(0, 0, 0), width=2)
    
    try:
        label_font = ImageFont.truetype(FONT_PATH, 12)
    except:
        label_font = ImageFont.load_default()
    
    label_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    label_bbox = draw.textbbox((0, 0), label, font=label_font)
    label_w = label_bbox[2] - label_bbox[0]
    label_h = label_bbox[3] - label_bbox[1]
    label_x = (size - label_w) // 2
    label_y = (size - label_h) // 2
    draw.text((label_x, label_y), label, fill=label_color, font=label_font)
    
    return img, icon_type, color


def generate_placement_scene():
    """Generate scene with labeled shapes. Questions identify by label, ask about color or shape."""
    img_w = random.randint(400, 1200)
    img_h = random.randint(300, 900)
    
    bg = generate_background(img_w, img_h)
    
    num_objects = random.randint(5, 10)
    objects = []
    used_labels = set()
    colors_list = list(COLORS)
    random.shuffle(colors_list)
    
    for i in range(num_objects):
        for _ in range(50):
            label = generate_label()
            if label not in used_labels:
                used_labels.add(label)
                break
        
        size = random.randint(50, 100)
        color = colors_list[i % len(colors_list)]
        
        element, shape_type, color = draw_labeled_icon(size, label, color=color)
        
        x = random.randint(0, max(0, img_w - size))
        y = random.randint(0, max(0, img_h - size))
        
        bg.paste(element, (x, y), element)
        
        objects.append({
            "label": label,
            "type": shape_type,
            "color": color,
            "color_name": rgb_to_name(color),
            "bbox": [x, y, x + size, y + size],
            "center": [x + size // 2, y + size // 2],
            "size": size
        })
    
    num_texts = random.randint(2, 4)
    words_sample = random.sample(WORDS, min(num_texts, len(WORDS)))
    
    for i, word in enumerate(words_sample):
        size = random.randint(40, 80)
        color = colors_list[(num_objects + i) % len(colors_list)]
        element, text, color = draw_text_element(size, word=word, color=color)
        
        x = random.randint(0, max(0, img_w - size))
        y = random.randint(0, max(0, img_h - size))
        
        bg.paste(element, (x, y), element)
        
        objects.append({
            "label": word,  # The word itself is the label
            "type": "text",
            "word": word,
            "color": color,
            "color_name": rgb_to_name(color),
            "bbox": [x, y, x + size, y + size],
            "center": [x + size // 2, y + size // 2],
            "size": size
        })
    
    target = random.choice(objects)
    
    if target["type"] == "text":
        # For text: identify by word, ask about color
        questions = [
            f"Zoom into the area where '{target['word']}' is written.",
            f"Crop around the word '{target['word']}' and inspect it.",
        ]
        problem = random.choice(questions) + " Put your final answer in <boxed></boxed> tags."
        answer = f"<boxed>{target['color_name']}</boxed>"
        ref_label = target['word']
    else:
        # For shapes: identify by label, ask about color OR shape
        ask_type = random.choice(['color', 'shape'])
        
        if ask_type == 'color':
            questions = [
                f"Zoom into the shape labeled '{target['label']}'.",
                f"Crop around the shape with label '{target['label']}' and inspect it.",
                f"Zoom into the {target['color_name']} {target['type']}.",
                f"Crop around the {target['color_name']} {target['type']}."
            ]
            answer = f"<boxed>{target['color_name']}</boxed>"
        else:
            questions = [
                f"Zoom into the shape labeled '{target['label']}'",
                f"Crop around the shape with label '{target['label']}' and inspect it.",
            ]
            answer = f"<boxed>{target['type']}</boxed>"
        
        problem = random.choice(questions) + " Put your final answer in <boxed></boxed> tags."
        ref_label = target['label']
    
    expected_tool_calls = [{"tool": "image_zoom_in_tool", "parameters": {"bbox_2d": target["bbox"], "label": ref_label}}]
    
    metadata = {
        "objects": objects,
        "target": target,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "zoom_placement",
        "img_dims": [img_w, img_h],
        "num_objects": len(objects)
    }
    
    return bg, problem, answer, expected_tool_calls, metadata


def generate_layered_scene():
    """Generate layered shapes with labels. Questions identify by label, ask about color or shape."""
    img_w = random.randint(500, 1000)
    img_h = random.randint(400, 800)
    
    bg = generate_background(img_w, img_h)
    
    num_layers = random.randint(4, 7)
    layers = []
    used_labels = set()
    colors_list = list(COLORS)
    random.shuffle(colors_list)
    
    for i in range(num_layers):
        # Generate unique label
        for _ in range(50):
            label = generate_label()
            if label not in used_labels:
                used_labels.add(label)
                break
        
        size = random.randint(80, min(img_w, img_h) // 2)
        color = colors_list[i % len(colors_list)]
        shape = random.choice(SHAPES)
        
        # Draw labeled shape
        layer_img, shape_type, color = draw_labeled_icon(size, label, color=color, shape_type=shape)
        
        x = random.randint(0, img_w - size)
        y = random.randint(0, img_h - size)
        
        bg.paste(layer_img, (x, y), layer_img)
        
        layers.append({
            "label": label,
            "shape": shape_type,
            "color": color,
            "color_name": rgb_to_name(color),
            "bbox": [x, y, x + size, y + size],
            "depth": i
        })
    
    target = random.choice(layers)
    
    # Ask about color OR shape (never both in question)
    ask_type = random.choice(['shape', 'color'])
    
    if ask_type == 'shape':
        questions = [
            f"Zoom into the shape labeled '{target['label']}'.",
            f"Crop around the shape with label '{target['label']}'.",
            f"Zoom into the {target['color_name']} {target['shape']}.",
            f"Crop around the {target['color_name']} {target['shape']}."
        ]
        answer = f"<boxed>{target['shape']}</boxed>"
    else:
        questions = [
            f"Zoom into the shape labeled '{target['label']}'.",
            f"Crop around the shape with label '{target['label']}'.",
        ]
        answer = f"<boxed>{target['color_name']}</boxed>"
    
    problem = random.choice(questions) + " Put your final answer in <boxed></boxed> tags."
    
    expected_tool_calls = [{"tool": "image_zoom_in_tool", "parameters": {"bbox_2d": target["bbox"], "label": target["label"]}}]
    
    metadata = {
        "layers": layers,
        "target": target,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "zoom_layered",
        "img_dims": [img_w, img_h]
    }
    
    return bg, problem, answer, expected_tool_calls, metadata
def generate_scattered_icons():
    """Generate scattered labeled icons. Questions identify by label, ask about color or shape."""
    img_w = random.randint(600, 1200)
    img_h = random.randint(400, 900)
    
    bg = generate_background(img_w, img_h)
    
    num_icons = random.randint(6, 12)
    icons = []
    used_labels = set()
    colors_list = list(COLORS)
    random.shuffle(colors_list)
    
    for i in range(num_icons):
        # Generate unique label
        for _ in range(50):
            label = generate_label()
            if label not in used_labels:
                used_labels.add(label)
                break
        
        size = random.randint(50, 80)
        color = colors_list[i % len(colors_list)]
        
        icon_img, icon_type, color = draw_labeled_icon(size, label, color=color)
        
        x = random.randint(10, img_w - size - 10)
        y = random.randint(10, img_h - size - 10)
        
        bg.paste(icon_img, (x, y), icon_img)
        
        icons.append({
            "label": label,
            "type": icon_type,
            "color": color,
            "color_name": rgb_to_name(color),
            "bbox": [x, y, x + size, y + size]
        })
    
    target = random.choice(icons)
    
    # Ask about color OR shape
    ask_type = random.choice(['color', 'shape'])
    
    if ask_type == 'color':
        questions = [
            f"Zoom into the shape labeled '{target['label']}'.",
            f"Crop around the icon with label '{target['label']}'.",
            f"Zoom into the {target['color_name']} {target['type']}.",
            f"Crop around the {target['color_name']} {target['type']}."
        ]
        answer = f"<boxed>{target['color_name']}</boxed>"
    else:
        questions = [
            f"Zoom into the shape labeled '{target['label']}'.",
            f"Crop around the icon with label '{target['label']}'.",
        ]
        answer = f"<boxed>{target['type']}</boxed>"
    
    problem = random.choice(questions) + " Put your final answer in <boxed></boxed> tags."
    
    expected_tool_calls = [{"tool": "image_zoom_in_tool", "parameters": {"bbox_2d": target["bbox"], "label": target["label"]}}]
    
    metadata = {
        "icons": icons,
        "target": target,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "zoom_icons",
        "img_dims": [img_w, img_h]
    }
    
    return bg, problem, answer, expected_tool_calls, metadata


def generate_quadrant_scene():
    """Generate quadrant scene with labeled shapes. Questions identify by label, ask about color or shape."""
    img_w = random.randint(400, 1000)
    img_h = random.randint(400, 800)
    
    bg = generate_background(img_w, img_h)
    draw = ImageDraw.Draw(bg)
    
    # Draw quadrant dividers
    draw.line([(img_w // 2, 0), (img_w // 2, img_h)], fill=(100, 100, 100), width=2)
    draw.line([(0, img_h // 2), (img_w, img_h // 2)], fill=(100, 100, 100), width=2)
    
    quadrants = {
        "top-left": (0, 0, img_w // 2, img_h // 2),
        "top-right": (img_w // 2, 0, img_w, img_h // 2),
        "bottom-left": (0, img_h // 2, img_w // 2, img_h),
        "bottom-right": (img_w // 2, img_h // 2, img_w, img_h)
    }
    
    placed = []
    used_labels = set()
    colors_list = list(COLORS)
    random.shuffle(colors_list)
    color_idx = 0
    
    for qname, (x1, y1, x2, y2) in quadrants.items():
        num_items = random.randint(2, 4)
        for j in range(num_items):
            # Generate unique label
            for _ in range(50):
                label = generate_label()
                if label not in used_labels:
                    used_labels.add(label)
                    break
            
            size = random.randint(40, 60)
            color = colors_list[color_idx % len(colors_list)]
            color_idx += 1
            
            icon_img, icon_type, color = draw_labeled_icon(size, label, color=color)
            
            x = random.randint(x1 + 10, max(x1 + 15, x2 - size - 10))
            y = random.randint(y1 + 10, max(y1 + 15, y2 - size - 10))
            
            bg.paste(icon_img, (x, y), icon_img)
            
            placed.append({
                "quadrant": qname,
                "label": label,
                "type": icon_type,
                "color": color,
                "color_name": rgb_to_name(color),
                "bbox": [x, y, x + size, y + size]
            })
    
    target = random.choice(placed)
    
    # Ask about color OR shape
    ask_type = random.choice(['color', 'shape'])
    
    if ask_type == 'color':
        questions = [
            f"Zoom into the shape labeled '{target['label']}'.",
            f"Crop around the shape with label '{target['label']}'.",
            f"Zoom into the {target['color_name']} {target['type']}.",
            f"Crop around the {target['color_name']} {target['type']}."
        ]
        answer = f"<boxed>{target['color_name']}</boxed>"
    else:
        questions = [
            f"Zoom into the shape labeled '{target['label']}'.",
            f"Crop around the shape with label '{target['label']}'.",
        ]
        answer = f"<boxed>{target['type']}</boxed>"
    
    problem = random.choice(questions) + " Put your final answer in <boxed></boxed> tags."
    
    expected_tool_calls = [{"tool": "image_zoom_in_tool", "parameters": {"bbox_2d": target["bbox"], "label": target["label"]}}]
    
    metadata = {
        "placed": placed,
        "quadrants": {k: list(v) for k, v in quadrants.items()},
        "target": target,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "zoom_quadrant",
        "img_dims": [img_w, img_h]
    }
    
    return bg, problem, answer, expected_tool_calls, metadata

def generate_scattered_labels():
    """Generate scattered text labels. Questions identify by word, ask about color."""
    img_w = random.randint(400, 1200)
    img_h = random.randint(300, 900)
    bg = generate_background(img_w, img_h)
    draw = ImageDraw.Draw(bg)
    
    num_labels = random.randint(8, 15)
    labels_list = []
    
    words = random.sample(WORDS, min(num_labels, len(WORDS)))
    colors_list = list(COLORS)
    random.shuffle(colors_list)
    
    for i in range(num_labels):
        x = random.randint(50, img_w - 150)
        y = random.randint(50, img_h - 50)
        word = words[i % len(words)]
        
        try:
            font_size = random.randint(16, 28)
            font = ImageFont.truetype(FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()
        
        text_color = colors_list[i % len(colors_list)]
        draw.text((x, y), word, fill=text_color, font=font)
        
        bbox = draw.textbbox((x, y), word, font=font)
        labels_list.append({
            "word": word,
            "pos": [x, y],
            "bbox": list(bbox),
            "color": text_color,
            "color_name": rgb_to_name(text_color)
        })
    
    target = random.choice(labels_list)
    
    # For text: identify by word, ask about color
    questions = [
        f"Zoom into the area where '{target['word']}' is written.",
        f"Crop around the word '{target['word']}' and inspect it.",
    ]
    problem = random.choice(questions) + " Put your final answer in <boxed></boxed> tags."
    answer = f"<boxed>{target['color_name']}</boxed>"
    
    pad = random.randint(10, 30)
    zoom_bbox = [max(0, target['bbox'][0] - pad), max(0, target['bbox'][1] - pad), 
                 min(img_w, target['bbox'][2] + pad), min(img_h, target['bbox'][3] + pad)]
    
    expected_tool_calls = [{"tool": "image_zoom_in_tool", "parameters": {"bbox_2d": zoom_bbox, "label": target['word']}}]
    
    metadata = {
        "labels": labels_list,
        "target": target,
        "expected_tool_calls": expected_tool_calls,
        "task_type": "zoom_label",
        "img_dims": [img_w, img_h]
    }
    
    return bg, problem, answer, expected_tool_calls, metadata


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
            "data_source": r.get("data_source", "crop_eval"),
            "reward_model": r.get("reward_model", {}),
            "agent_name": r.get("agent_name", "tool_agent"),
            "tools_kwargs": r.get("tools_kwargs", {}),
            "available_tools": r.get("available_tools", ["image_zoom_in_tool"]),
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
    parser = argparse.ArgumentParser(description="Generate zoom/crop-only dataset")
    parser.add_argument("--output_dir", type=str, default="zoom", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=200, help="Total samples")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    generators = [
        generate_placement_scene,
        generate_layered_scene,
        generate_scattered_icons,
        generate_quadrant_scene,
        generate_scattered_labels,
    ]

    print(f"Generating {args.num_samples} samples...")
    rows = []
    for i in tqdm(range(args.num_samples), desc="Generating"):
        gen = generators[i % len(generators)]
        img, problem, answer, expected_tool_calls, metadata = gen()

        img_name = f"zoom_{i:05d}.png"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        img_abs = os.path.abspath(img_path)

        user_content = f"<image>\n{problem}\n\nUse the image_zoom_in_tool to inspect the relevant region closely, then provide your answer."
        
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tools_kwargs = {
            "image_zoom_in_tool": {"create_kwargs": {"image": img_abs}}
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
            "data_source": "crop_eval",
            "reward_model": reward_model,
            "agent_name": "tool_agent",
            "tools_kwargs": tools_kwargs,
            "available_tools": ["image_zoom_in_tool"],
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
    print(f"  Images dir: {images_dir}")

    try:
        df = pd.read_parquet(train_path, engine="pyarrow")
        print("Columns:", df.columns.tolist())
        counts = Counter([m.get("task_type", "unknown") for m in df["metadata"]])
        print("Task distribution:")
        for t, c in counts.items():
            print(f"  {t}: {c}")
    except Exception as e:
        print("Verification failed:", e)


if __name__ == "__main__":
    main()
