"""Pick candidate sequences for the fusion/robustness study.

Selection criteria, in priority order:
  1. Day and night sequences should share a route, so the day->night comparison
     isolates lighting rather than confounding it with a different location.
     Route proximity is approximated by GPS bounding-box overlap (IoU).
  2. Prefer good gps_quality (RTK_fixed_cm > float_dm > single_m) since GPS/CAN
     supervise the steering probe.
  3. Prefer full-length runs (~10 min) with a low idle_fraction - idle frames carry
     no steering signal and inflate the zero bin.
  4. Include degraded sequences (only 4 exist, all in the official test split).

Run:  .venv/Scripts/python.exe scripts/select_sequences.py
"""
import pandas as pd
from huggingface_hub import hf_hub_download

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

df = pd.read_parquet(hf_hub_download("anthonytec2/OctoSense", "metadata.parquet",
                                     repo_type="dataset"))
car = df[df["platform"] == "car"].copy()
for c in ("is_daytime", "degraded", "has_seg"):
    car[c] = car[c].astype(bool)


def bbox_iou(a, b):
    """IoU of two GPS bounding boxes; a crude 'same route?' proxy."""
    lat1 = max(a.gps_lat_min, b.gps_lat_min)
    lat2 = min(a.gps_lat_max, b.gps_lat_max)
    lon1 = max(a.gps_lon_min, b.gps_lon_min)
    lon2 = min(a.gps_lon_max, b.gps_lon_max)
    if lat2 <= lat1 or lon2 <= lon1:
        return 0.0
    inter = (lat2 - lat1) * (lon2 - lon1)
    area_a = (a.gps_lat_max - a.gps_lat_min) * (a.gps_lon_max - a.gps_lon_min)
    area_b = (b.gps_lat_max - b.gps_lat_min) * (b.gps_lon_max - b.gps_lon_min)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


print("=== sessions containing BOTH day and night ===")
mix = car.groupby("session")["is_daytime"].agg(["sum", "count"])
mix["n_night"] = mix["count"] - mix["sum"]
mix = mix.rename(columns={"sum": "n_day", "count": "n_total"})
print(mix[(mix.n_day > 0) & (mix.n_night > 0)].to_string())

night = car[~car["is_daytime"]]
day = car[car["is_daytime"]]

# For each night sequence, find its best route-matched daytime partner.
print("\n=== best route-matched day partner for each night sequence (top 12 by IoU) ===")
pairs = []
for _, n in night.iterrows():
    best, best_iou = None, 0.0
    for _, d in day.iterrows():
        i = bbox_iou(n, d)
        if i > best_iou:
            best, best_iou = d, i
    if best is not None:
        pairs.append({
            "night_bag": n.bag_id, "night_sess": n.session, "night_gps": n.gps_quality,
            "night_idle": round(float(n.idle_fraction), 3), "night_dur": n.duration_s,
            "night_degraded": bool(n.degraded),
            "day_bag": best.bag_id, "day_sess": best.session, "day_gps": best.gps_quality,
            "day_idle": round(float(best.idle_fraction), 3), "day_dur": best.duration_s,
            "route_iou": round(best_iou, 3),
        })

pdf = pd.DataFrame(pairs).sort_values("route_iou", ascending=False)
print(pdf.head(12).to_string(index=False))

print("\n=== the 4 degraded sequences (all official test split) ===")
cols = ["bag_id", "session", "is_daytime", "has_seg", "gps_quality",
        "duration_s", "idle_fraction", "mean_speed_mph", "distance_m", "split"]
print(car[car["degraded"]][cols].to_string(index=False))

print("\n=== route IoU of each degraded seq vs its best daytime partner ===")
for _, g in car[car["degraded"]].iterrows():
    best, best_iou = None, 0.0
    for _, d in day.iterrows():
        if d.bag_id == g.bag_id:
            continue
        i = bbox_iou(g, d)
        if i > best_iou:
            best, best_iou = d, i
    print(f"  {g.bag_id}  (day={g.is_daytime})  -> {best.bag_id}  IoU={best_iou:.3f}")
