#!/usr/bin/env python3
"""
Generate real-world bbox/crop dataset from RefCOCO for VERL training.
RefCOCO contains natural images with referring expressions and bounding boxes.
"""

import os
import sys
import random
import argparse
import json
import pickle
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from collections import Counter

# ==========================
# Configuration
# ==========================

# RefCOCO dataset URLs - try HuggingFace mirror first, then web archive
REFCOCO_URLS = {
    "refcoco": [
        "https://huggingface.co/datasets/jxu124/refcoco/resolve/main/refcoco.zip",
        "https://web.archive.org/web/20220413011718if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
    ],
    "refcoco+": [
        "https://huggingface.co/datasets/jxu124/refcoco/resolve/main/refcoco+.zip",
        "https://web.archive.org/web/20220413011656if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
    ],
    "refcocog": [
        "https://huggingface.co/datasets/jxu124/refcoco/resolve/main/refcocog.zip",
        "https://web.archive.org/web/20220413012904if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    ],
}

# COCO 2014 train images (used by RefCOCO)
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"

# COCO 2014 annotations for category names
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

SYSTEM_PROMPT_BBOX = """You are a helpful assistant that analyzes images using the bounding box tool.

When asked to locate objects in an image:
1. Use image_bbox_tool to draw a bounding box around the target object
2. After receiving the tool response, provide your final answer in <boxed></boxed> tags

Always use the bounding box tool to mark the relevant region."""

SYSTEM_PROMPT_CROP = """You are a helpful assistant that analyzes images using the crop tool.

When asked to focus on specific objects in an image:
1. Use image_crop_tool to crop the target region from the image
2. After receiving the tool response, provide your final answer in <boxed></boxed> tags

Always use the crop tool to isolate the relevant region."""



def download_file(url, dest_path, desc=None):
    """Download a file with progress bar."""
    if os.path.exists(dest_path):
        print(f"  [✓] {dest_path} already exists, skipping download")
        return dest_path
    
    print(f"  Downloading {desc or url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=300) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            with open(dest_path, 'wb') as f:
                downloaded = 0
                block_size = 8192
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    while True:
                        block = response.read(block_size)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)
                        pbar.update(len(block))
        return dest_path
    except Exception as e:
        print(f"  [!] Download failed: {e}")
        if os.path.exists(dest_path): os.remove(dest_path)
        raise


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f" Extracted to {extract_to}")


def download_and_extract_refcoco(data_dir, dataset="refcoco"):
    """Download and extract RefCOCO dataset."""
    urls = REFCOCO_URLS.get(dataset)
    if not urls:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    zip_path = os.path.join(data_dir, "downloads", f"{dataset}.zip")
    extract_dir = os.path.join(data_dir, "refer_data")
    
    if not os.path.exists(os.path.join(extract_dir, dataset)):
        # Try each URL until one works
        for url in urls:
            try:
                download_file(url, zip_path, desc=f"{dataset}.zip")
                extract_zip(zip_path, extract_dir)
                break
            except Exception as e:
                print(f" Failed with {url}: {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                continue
        else: raise RuntimeError(f"Failed to download {dataset} from all URLs")
    else: print(f" {dataset} already extracted")
    
    return os.path.join(extract_dir, dataset)


def download_coco_images(data_dir):
    """Download COCO 2014 train images."""
    zip_path = os.path.join(data_dir, "downloads", "train2014.zip")
    extract_dir = os.path.join(data_dir, "coco_images")
    images_dir = os.path.join(extract_dir, "train2014")
    
    if not os.path.exists(images_dir):
        download_file(COCO_IMAGES_URL, zip_path, desc="COCO train2014.zip")
        extract_zip(zip_path, extract_dir)
    else:
        print(f" COCO images already extracted")
    
    return images_dir


def download_coco_annotations(data_dir):
    """Download COCO 2014 annotations for category names."""
    zip_path = os.path.join(data_dir, "downloads", "annotations_trainval2014.zip")
    extract_dir = os.path.join(data_dir, "coco_annotations")
    instances_file = os.path.join(extract_dir, "annotations", "instances_train2014.json")
    
    if not os.path.exists(instances_file):
        download_file(COCO_ANNOTATIONS_URL, zip_path, desc="COCO annotations")
        extract_zip(zip_path, extract_dir)
    else:
        print(f" COCO annotations already extracted")
    
    return instances_file


class RefCOCOLoader:
    """Load RefCOCO dataset with referring expressions and bounding boxes."""
    
    def __init__(self, data_dir, dataset="refcoco", splitBy="unc"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.splitBy = splitBy
        
        ref_file = os.path.join(data_dir, f"refs({splitBy}).p")
        print(f"  Loading refs from {ref_file}...")
        with open(ref_file, 'rb') as f:
            self.refs = pickle.load(f)
        
        instances_file = os.path.join(data_dir, "instances.json")
        print(f"  Loading instances from {instances_file}...")
        with open(instances_file, 'r') as f:
            instances = json.load(f)
        
        self.images = {img['id']: img for img in instances['images']}
        self.annotations = {ann['id']: ann for ann in instances['annotations']}
        self.categories = {cat['id']: cat['name'] for cat in instances['categories']}
        
        self.ref_to_ann = {}
        for ref in self.refs:
            self.ref_to_ann[ref['ref_id']] = ref['ann_id']
        

        self.image_category_to_bboxes = {}
        for ann in instances['annotations']:
            key = (ann['image_id'], ann['category_id'])
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                bbox_xyxy = [int(x), int(y), int(x + w), int(y + h)]
                if key not in self.image_category_to_bboxes:
                    self.image_category_to_bboxes[key] = []
                self.image_category_to_bboxes[key].append(bbox_xyxy)
        
        print(f"  Loaded {len(self.refs)} refs, {len(self.images)} images, {len(self.annotations)} annotations")
    
    def get_ref_data(self, ref):
        """Get full data for a reference."""
        ann_id = ref['ann_id']
        ann = self.annotations.get(ann_id)
        if not ann:
            return None
        
        image_id = ref['image_id']
        image_info = self.images.get(image_id)
        if not image_info:
            return None
        
        category_id = ann.get('category_id')
        category_name = self.categories.get(category_id, "object")
        
        sentences = [s['raw'] for s in ref.get('sentences', [])]
        
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            return None
        
        x, y, w, h = bbox
        bbox_xyxy = [int(x), int(y), int(x + w), int(y + h)]
        

        key = (image_id, category_id)
        all_valid_bboxes = self.image_category_to_bboxes.get(key, [bbox_xyxy])
        
        return {
            'ref_id': ref['ref_id'],
            'ann_id': ann_id,
            'image_id': image_id,
            'image_file': image_info['file_name'],
            'image_width': image_info['width'],
            'image_height': image_info['height'],
            'category': category_name,
            'category_id': category_id,
            'sentences': sentences,
            'bbox': bbox_xyxy, 
            'all_valid_bboxes': all_valid_bboxes,  
            'bbox_xywh': bbox,
            'split': ref.get('split', 'train'),
        }
    
    def get_split(self, split='train'):
        """Get all refs for a specific split."""
        return [ref for ref in self.refs if ref.get('split') == split]



def generate_bbox_prompt(ref_data, referring_expr):
    """Generate a bbox task prompt."""
    templates = [
        f"Use the bounding box tool to locate: {referring_expr}",
        f"Draw a bounding box around {referring_expr}",
        f"Find and mark with a bbox: {referring_expr}",
        f"Locate {referring_expr} in this image using the bbox tool",
        f"Use image_bbox_tool to highlight: {referring_expr}",
        f"Identify the location of {referring_expr} with a bounding box",
    ]
    return random.choice(templates)

def generate_crop_prompt(ref_data, referring_expr):
    """Generate a crop task prompt."""
    templates = [
        f"Use the crop tool to focus on: {referring_expr}",
        f"Crop the region showing {referring_expr}",
        f"Extract the area containing: {referring_expr}",
        f"Use image_crop_tool to isolate: {referring_expr}",
        f"Crop out {referring_expr} from this image",
    ]
    return random.choice(templates)



def create_bbox_sample(ref_data, images_dir, task_type="bbox"):
    """Create a training sample for bbox or crop task."""
    image_path = os.path.join(images_dir, ref_data['image_file'])
    if not os.path.exists(image_path): return None
    if not ref_data['sentences']: return None
    referring_expr = random.choice(ref_data['sentences'])
    
    if task_type == "bbox":
        problem = generate_bbox_prompt(ref_data, referring_expr)
        system_prompt = SYSTEM_PROMPT_BBOX
        tool_name = "image_bbox_tool"
    else:
        problem = generate_crop_prompt(ref_data, referring_expr)
        system_prompt = SYSTEM_PROMPT_CROP
        tool_name = "image_crop_tool"
    
    ground_truth_bbox = ref_data.get('all_valid_bboxes', [ref_data['bbox']])
    
    img_abs = os.path.abspath(image_path)
    user_content = f"<image>\n{problem}\n\nProvide the bounding box coordinates."
    
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    tools_kwargs = {tool_name: {"create_kwargs": {"image": img_abs}}}

    reward_model = {
        "style": "rule",
        "ground_truth_bbox": ground_truth_bbox, 
        "ground_truth": ground_truth_bbox,
        "type": "refcoco_iou",
    }

    metadata = {
        "ref_id": ref_data['ref_id'],
        "image_id": ref_data['image_id'],
        "category": ref_data['category'],
        "referring_expression": referring_expr,
        "ground_truth_bbox": ground_truth_bbox,
        "task_type": task_type,
    }

    extra_info = {
        "expected_tool_calls": [{"tool": tool_name, "parameters": {"bbox_2d": ground_truth_bbox[0]}}],
        "metadata": metadata,
        "tools_kwargs": tools_kwargs,
    }

    return {
        "prompt": prompt,
        # For refcoco we don't require a boxed textual answer — reward is IoU/tool-use based
        "answer": "",
        "images": [img_abs],
        "metadata": metadata,
        "data_source": f"refcoco_{task_type}",
        "reward_model": reward_model,
        "tools_kwargs": tools_kwargs,
        "extra_info": extra_info,
        "agent_name": "tool_agent",
        "available_tools": [tool_name],
    }


def save_as_parquet(rows, output_dir, split_name="train"):
    """Save dataset as VERL parquet format."""
    os.makedirs(output_dir, exist_ok=True)
    
    pa_rows = []
    for r in rows:
        pa_rows.append({
            "prompt": r["prompt"],
            "answer": r["answer"],
            "images": r["images"],
            "metadata": r.get("metadata", {}),
            "extra_info": r.get("extra_info", {}),
            "data_source": r.get("data_source", "refcoco"),
            "reward_model": r.get("reward_model", {}),
            "tools_kwargs": r.get("tools_kwargs", {}),
            "available_tools": r.get("available_tools", ["image_bbox_tool"]),
            "agent_name": r.get("agent_name", "tool_agent"),
        })
    
    table = pa.Table.from_pylist(pa_rows)
    out_path = os.path.join(output_dir, f"{split_name}.parquet")
    pq.write_table(table, out_path)
    return out_path, len(pa_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate RefCOCO bbox/crop dataset for VERL")
    parser.add_argument("--data_dir", type=str, default="/users/$USER/$SCRATCH/data/refcoco_raw", help="Directory to store raw data")
    parser.add_argument("--output_dir", type=str, default="data/refcoco", help="Output directory for parquet files")
    parser.add_argument("--dataset", type=str, default="refcoco", choices=["refcoco", "refcoco+", "refcocog"], help="RefCOCO dataset variant")
    parser.add_argument("--split_by", type=str, default="unc", help="Split type (unc, google, umd)")
    parser.add_argument("--task_type", type=str, default="both", choices=["bbox", "crop", "both"], help="Task type to generate")
    parser.add_argument("--max_samples", type=int, default=4000, help="Maximum samples for training (default: 4000)")
    parser.add_argument("--test_samples", type=int, default=500, help="Maximum samples for testing (default: 500)")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading (use existing data)")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    

    
    if not args.skip_download:
        print(f"\nDownloading {args.dataset}...")
        refcoco_dir = download_and_extract_refcoco(args.data_dir, args.dataset)
        
        print("\nDownloading COCO 2014 train images...")
        coco_images_dir = download_coco_images(args.data_dir)
    else:
        refcoco_dir = os.path.join(args.data_dir, "refer_data", args.dataset)
        coco_images_dir = os.path.join(args.data_dir, "coco_images", "train2014")
        print(f"  RefCOCO: {refcoco_dir}")
        print(f"  COCO images: {coco_images_dir}")
    
    if not os.path.exists(refcoco_dir):
        print(f"\n[ERROR] RefCOCO data not found at {refcoco_dir}")
        print("Please run without --skip_download to download the data first.")
        sys.exit(1)
    
    if not os.path.exists(coco_images_dir):
        print(f"\n[ERROR] COCO images not found at {coco_images_dir}")
        print("Please run without --skip_download to download the images first.")
        sys.exit(1)
    
    loader = RefCOCOLoader(refcoco_dir, args.dataset, args.split_by)
    
    task_types = ["bbox", "crop"] if args.task_type == "both" else [args.task_type]
    
    for task_type in task_types:
        print(f"\n--- Generating {task_type.upper()} samples ---")
        
        all_rows = []
        
        train_refs = loader.get_split('train')
        print(f"  Found {len(train_refs)} train refs")
        
        random.shuffle(train_refs)
        if args.max_samples:
            train_refs = train_refs[:args.max_samples]
            print(f"  Limited to {len(train_refs)} train samples")
        
        for ref in tqdm(train_refs, desc=f"Processing {task_type}"):
            ref_data = loader.get_ref_data(ref)
            if ref_data is None:
                continue
            
            sample = create_bbox_sample(ref_data, coco_images_dir, task_type)
            if sample:
                all_rows.append(sample)
        
        val_refs = loader.get_split('val')
        print(f"  Found {len(val_refs)} val refs")
        
        random.shuffle(val_refs)
        if args.test_samples:
            val_refs = val_refs[:args.test_samples]
            print(f"  Limited to {len(val_refs)} test samples")
        
        test_rows = []
        for ref in tqdm(val_refs, desc=f"Processing {task_type} (val)"):
            ref_data = loader.get_ref_data(ref)
            if ref_data is None:
                continue
            
            sample = create_bbox_sample(ref_data, coco_images_dir, task_type)
            if sample:
                test_rows.append(sample)
        
        print(f"\n  Generated {len(all_rows)} train samples, {len(test_rows)} test samples")
        
        task_output_dir = os.path.join(args.output_dir, task_type)
        
        train_path, train_count = save_as_parquet(all_rows, task_output_dir, "train")
        test_path, test_count = save_as_parquet(test_rows, task_output_dir, "test")
        
        print(f"    Train ({train_count}): {train_path}")
        print(f"    Test  ({test_count}): {test_path}")
        
        try:
            df = pd.read_parquet(train_path, engine="pyarrow")
            print(f"    Columns: {df.columns.tolist()}")
            
            categories = [m.get('category', 'unknown') for m in df['metadata']]
            cat_counts = Counter(categories)
            print("    Top 10 categories:")
            for cat, count in cat_counts.most_common(10):
                print(f"      {cat}: {count}")
            
            num_valid = [len(m.get('ground_truth_bbox', [1])) for m in df['metadata']]
            multi_bbox_count = sum(1 for n in num_valid if n > 1)
            print(f"    Samples with multiple valid bboxes: {multi_bbox_count}/{len(df)} ({100*multi_bbox_count/len(df):.1f}%)")
        except Exception as e:
            print(f"    Verification failed: {e}")
    

    print(f"\nOutput directory: {args.output_dir}")
    if "bbox" in task_types:
        print(f"  train_files: {os.path.join(args.output_dir, 'bbox', 'train.parquet')}")
        print(f"  val_files: {os.path.join(args.output_dir, 'bbox', 'test.parquet')}")
    if "crop" in task_types:
        print(f"  train_files: {os.path.join(args.output_dir, 'crop', 'train.parquet')}")
        print(f"  val_files: {os.path.join(args.output_dir, 'crop', 'test.parquet')}")


if __name__ == "__main__":
    main()
