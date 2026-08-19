"""Verify which ground-truth files actually exist on the hub for day vs night car
sequences. metadata.parquet flags has_seg but has no depth flag, so check the
real file list rather than infer.

Run:  .venv/Scripts/python.exe scripts/check_gt_files.py
"""
import re
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "anthonytec2/OctoSense"

meta = pd.read_parquet(hf_hub_download(REPO, "metadata.parquet", repo_type="dataset"))
car = meta[meta["platform"] == "car"].copy()
car["is_daytime"] = car["is_daytime"].astype(bool)
day_flag = dict(zip(car["bag_id"], car["is_daytime"]))

files = list_repo_files(REPO, repo_type="dataset")

# car/<session>/<bag_id>/<file>
per_bag = defaultdict(set)
for f in files:
    m = re.match(r"^car/[^/]+/([^/]+)/(.+)$", f)
    if m:
        per_bag[m.group(1)].add(m.group(2))

print(f"car sequences with files on hub: {len(per_bag)}\n")

rows = []
for bag, fs in per_bag.items():
    if bag not in day_flag:
        continue
    rows.append({
        "bag_id": bag,
        "is_daytime": day_flag[bag],
        "data.h5": "data.h5" in fs,
        "depth": "rgb_left_rect_depth.h5" in fs,
        "seg": "rgb_left_rect_semantic.h5" in fs,
        "img_left": "img_left.mp4" in fs,
        "img_right": "img_right.mp4" in fs,
        "ir": "img_infrared.mp4" in fs,
    })

df = pd.DataFrame(rows)
print("=== GT/file availability by time of day (counts of True) ===")
summary = df.groupby("is_daytime")[
    ["data.h5", "depth", "seg", "img_left", "img_right", "ir"]
].sum()
summary["n_sequences"] = df.groupby("is_daytime").size()
print(summary.to_string())

print("\n=== as percentages ===")
pct = summary.drop(columns="n_sequences").div(summary["n_sequences"], axis=0).mul(100).round(1)
print(pct.to_string())

print("\n=== sample of NIGHT sequences and their files ===")
print(df[~df["is_daytime"]].head(8).to_string(index=False))
