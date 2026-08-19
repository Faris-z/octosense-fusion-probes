"""Check ground-truth availability against condition, which determines what the
Phase 6 robustness comparison can actually measure.

Note two quirks in metadata.parquet:
  - is_daytime / degraded / has_seg are object dtype holding Python bools, so `~col`
    raises; coerce with .astype(bool) first.
  - sensor_dropout uses the literal string "nan" rather than a real null.

Run:  .venv/Scripts/python.exe scripts/analyze_availability.py
"""
import pandas as pd
from huggingface_hub import hf_hub_download

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_parquet(hf_hub_download("anthonytec2/OctoSense", "metadata.parquet",
                                     repo_type="dataset"))
car = df[df["platform"] == "car"].copy()

for c in ("is_daytime", "degraded", "has_seg"):
    car[c] = car[c].astype(bool)
car["has_dropout"] = car["sensor_dropout"].astype(str).ne("nan")

print("=== COLUMNS ===")
print(list(car.columns))

print("\n=== has_seg vs is_daytime ===")
print(pd.crosstab(car["is_daytime"], car["has_seg"], margins=True).to_string())

depth_cols = [c for c in car.columns if "depth" in c.lower()]
print(f"\ndepth-availability columns: {depth_cols or 'NONE - depth availability not flagged'}")

print("\n=== sequences with a real sensor dropout ===")
show = [c for c in ("bag_id", "session", "is_daytime", "has_seg", "degraded",
                    "sensor_dropout", "gps_quality", "duration_s", "split")
        if c in car.columns]
print(car[car["has_dropout"]][show].to_string(index=False))

print("\n=== degraded == True ===")
print(car[car["degraded"]][show].to_string(index=False))

night = car[~car["is_daytime"]]
print(f"\n=== NIGHT sequences: {len(night)} ===")
print(f"  has_seg True: {int(night['has_seg'].sum())} / {len(night)}")
print("  gps_quality:\n" + night["gps_quality"].value_counts().to_string())
print("  split:\n" + night["split"].value_counts().to_string())
print("  sessions:\n" + night["session"].value_counts().to_string())

print("\n=== official split vs condition ===")
print(pd.crosstab(car["is_daytime"], car["split"], margins=True).to_string())
