#!/usr/bin/env python3
"""Pretokenize vision data in parquet files for efficient VLM training."""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def load_vision_tokenizer(model_name: str, device: str = "cuda"):
    """Load the appropriate vision tokenizer."""
    if "qwen" in model_name.lower():
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_name)
        from verl.utils.vision_tokenizer import QwenVLTokenizer
        return QwenVLTokenizer(model, processor)
    
    elif "emu3" in model_name.lower():
        from transformers import AutoModel, AutoImageProcessor
        tokenizer = AutoModel.from_pretrained(
            "BAAI/Emu3-VisionTokenizer", trust_remote_code=True, device_map=device
        ).eval()
        processor = AutoImageProcessor.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True)
        from verl.utils.vision_tokenizer import Emu3Tokenizer
        return Emu3Tokenizer(tokenizer, processor)
    
    raise ValueError(f"Unsupported model: {model_name}")


def extract_images_from_messages(messages: list) -> list:
    """Extract image paths from chat messages."""
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                img_path = item.get("image")
                if img_path:
                    images.append(img_path)
    return images


def pretokenize_parquet(input_path: str, output_path: str, model_name: str, device: str = "cuda"):
    """Pretokenize all images in a parquet file."""
    print(f"Loading vision tokenizer: {model_name}")
    tokenizer = load_vision_tokenizer(model_name, device)
    
    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)
    
    vision_cache = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        prompt = row.get("prompt", [])
        images = extract_images_from_messages(prompt)
        
        cached_images = []
        for img_path in images:
            try:
                image = Image.open(img_path).convert("RGB")
                encoded = tokenizer.encode(image)
                # Convert numpy arrays to lists for JSON serialization in parquet
                for k, v in encoded.items():
                    if isinstance(v, np.ndarray):
                        encoded[k] = v.tolist()
                cached_images.append(encoded)
            except Exception as e:
                print(f"Warning: Failed to encode {img_path}: {e}")
                cached_images.append(None)
        
        vision_cache.append({
            "token_type": tokenizer.token_type,
            "images": cached_images if cached_images else None,
        })
    
    # Add vision cache to extra_info
    extra_info_col = df.get("extra_info", [{}] * len(df))
    for i, cache in enumerate(vision_cache):
        if cache["images"]:
            if isinstance(extra_info_col.iloc[i], dict):
                extra_info_col.iloc[i]["vision_cache"] = cache
            else:
                extra_info_col.iloc[i] = {"vision_cache": cache}
    
    df["extra_info"] = extra_info_col
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pretokenize vision data in parquet files")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    
    pretokenize_parquet(args.input, args.output, args.model, args.device)


if __name__ == "__main__":
    main()
