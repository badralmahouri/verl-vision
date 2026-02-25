#!/usr/bin/env python3
"""
Fix existing mixed_tools parquet files by ensuring `reward_model` contains
a `ground_truth` entry. The script updates train.parquet and test.parquet
under the provided output directory.

Usage:
  python data/fix_mixed_tools_reward_model.py --parquet_dir data/refcoco_mixed/mixed_tools
"""
import os
import argparse
import pyarrow.parquet as pq
import pyarrow as pa


def extract_gt_from_row(row):
    # Try expected_tool_calls first
    extra = row.get("extra_info") or {}
    calls = extra.get("expected_tool_calls") or []
    for c in calls:
        if c.get("tool") == "image_bbox_tool":
            params = c.get("parameters") or {}
            bbox = params.get("bbox_2d") or params.get("bbox")
            if bbox and len(bbox) == 4:
                return [list(map(int, bbox))]
    # Fallback to metadata
    md = row.get("metadata") or {}
    gt = md.get("ground_truth_bbox") or md.get("gt_bbox")
    if gt and len(gt) == 4:
        return [list(map(int, gt))]
    return []


def fix_file(path):
    print("Processing", path)
    table = pq.read_table(path)
    rows = table.to_pylist()
    changed = 0
    for r in rows:
        rm = r.get("reward_model")
        if not isinstance(rm, dict) or "ground_truth" not in rm:
            gt = extract_gt_from_row(r)
            new_rm = rm if isinstance(rm, dict) else {}
            new_rm["ground_truth"] = gt
            # If bbox present and gt is non-empty, add refcoco_iou metadata
            if gt and "type" not in new_rm:
                new_rm.setdefault("style", "rule")
                new_rm.setdefault("type", "refcoco_iou")
                new_rm.setdefault("ground_truth_bbox", gt)
            r["reward_model"] = new_rm
            changed += 1

    if changed == 0:
        print("No changes needed.")
        return

    out_table = pa.Table.from_pylist(rows)
    backup = path + ".bak"
    try:
        os.rename(path, backup)
    except Exception:
        pass
    pq.write_table(out_table, path)
    print(f"Updated {changed} rows (backup at {backup}).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", type=str, required=True)
    args = parser.parse_args()

    for name in ("train.parquet", "test.parquet"):
        p = os.path.join(args.parquet_dir, name)
        if os.path.exists(p):
            fix_file(p)
        else:
            print("Missing", p)


if __name__ == "__main__":
    main()
