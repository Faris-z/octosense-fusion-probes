"""Estimate download size for the candidate sequences, per modality.

Run:  .venv/Scripts/python.exe scripts/estimate_size.py
"""
import re
from collections import defaultdict

from huggingface_hub import HfApi

REPO = "anthonytec2/OctoSense"

CANDIDATES = [
    "rosbag2_2026_01_04-14_12_18",  # day   - route partner of 01_04-20_33_11
    "rosbag2_2026_01_14-08_22_52",  # day   - route partner of 01_15-20_09_30
    "rosbag2_2026_01_04-20_33_11",  # night - IoU 0.982 vs 01_04-14_12_18
    "rosbag2_2026_01_15-20_09_30",  # night - IoU 0.944 vs 01_14-08_22_52
    "rosbag2_2026_01_08-20_47_06",  # night + degraded
    "rosbag2_2026_01_09-07_47_28",  # day   + degraded
]

WANTED = {
    "data.h5": "data",
    "rgb_left_rect_depth.h5": "depth",
    "rgb_left_rect_semantic.h5": "seg",
    "img_left.mp4": "rgb_left",
    "img_right.mp4": "rgb_right",
    "img_infrared.mp4": "ir",
    "events.h5": "events (SKIPPED)",
}

api = HfApi()
info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)

sizes = defaultdict(dict)
for sib in info.siblings:
    m = re.match(r"^car/[^/]+/([^/]+)/(.+)$", sib.rfilename)
    if m and m.group(1) in CANDIDATES and m.group(2) in WANTED:
        sizes[m.group(1)][WANTED[m.group(2)]] = (sib.size or 0) / 1e9

GB = lambda x: f"{x:6.2f}"
keep = ["data", "depth", "seg", "rgb_left", "rgb_right", "ir"]

print(f"{'sequence':<30} " + " ".join(f"{k:>10}" for k in keep) + f" {'KEEP':>8} {'events':>8}")
print("-" * 110)
tot_keep = tot_ev = 0.0
for bag in CANDIDATES:
    s = sizes.get(bag, {})
    row_keep = sum(s.get(k, 0) for k in keep)
    ev = s.get("events (SKIPPED)", 0)
    tot_keep += row_keep
    tot_ev += ev
    print(f"{bag:<30} " + " ".join(GB(s.get(k, 0)) + "    " for k in keep)
          + f" {GB(row_keep)}  {GB(ev)}")

print("-" * 110)
print(f"{'TOTAL (GB)':<30} " + " " * (11 * len(keep)) + f" {GB(tot_keep)}  {GB(tot_ev)}")
print(f"\nDownloading {len(CANDIDATES)} sequences without events: ~{tot_keep:.1f} GB")
print(f"Skipping events saves: ~{tot_ev:.1f} GB "
      f"({100 * tot_ev / (tot_ev + tot_keep):.0f}% of full size)")
print(f"\nIf you later drop img_right (mono instead of stereo), saves another "
      f"~{sum(sizes[b].get('rgb_right', 0) for b in sizes):.1f} GB")
