"""Download OctoSense metadata.parquet and print its schema + a condition summary.

This is metadata only (one row per sequence), not sensor payload - a few hundred KB.

Run:  .venv/Scripts/python.exe scripts/fetch_metadata.py
"""
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "anthonytec2/OctoSense"

path = hf_hub_download(REPO, "metadata.parquet", repo_type="dataset")
df = pd.read_parquet(path)

print(f"downloaded -> {path}")
print(f"shape: {df.shape}\n")

print("--- columns / dtypes ---")
for col, dt in df.dtypes.items():
    print(f"  {col:<22} {dt}")

print("\n--- platform counts ---")
print(df["platform"].value_counts().to_string())

car = df[df["platform"] == "car"]
print(f"\n--- car sequences: {len(car)} ---")

for col in ("is_daytime", "degraded", "has_seg", "gps_quality",
            "sensor_dropout", "session", "split"):
    if col in car.columns:
        print(f"\n{col}:")
        print(car[col].value_counts(dropna=False).to_string())

print("\n--- day/night x degraded crosstab (car) ---")
if {"is_daytime", "degraded"}.issubset(car.columns):
    print(pd.crosstab(car["is_daytime"], car["degraded"]).to_string())

print("\n--- duration / distance stats (car) ---")
stats = [c for c in ("duration_s", "distance_m", "mean_speed_mph",
                     "idle_fraction", "n_rgb_frames") if c in car.columns]
print(car[stats].describe().to_string())
