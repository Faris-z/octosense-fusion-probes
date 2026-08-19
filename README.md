# octosense-fusion-probes

A learning project exploring **multi-sensor fusion and robustness** (day vs. night / degraded sensors)
on the [OctoSense dataset](https://huggingface.co/datasets/anthonytec2/OctoSense) — stereo RGB, event,
LiDAR, thermal, IMU, RTK-GPS, and CAN bus from a shared sensor platform.

This is **not** a reproduction of the OctoSense paper's masked-autoencoder model (its training code
hasn't been released). Instead it's a smaller, laptop/PC-friendly pipeline:

> **frozen per-modality feature extractors → lightweight probes** for depth, segmentation, and
> ego-motion/steering, evaluated for robustness across lighting and sensor conditions.

The core question: **does fusion help more when a sensor degrades?**

## References

| | |
|---|---|
| Tooling repo | https://github.com/anthonytec2/OctoSense |
| Dataset (HF) | https://huggingface.co/datasets/anthonytec2/OctoSense |
| Paper | https://arxiv.org/abs/2606.27317 |

## Machines

The work splits across two machines, deliberately:

- **Laptop (Phases 0–3)** — environment, sequence selection, download, cleaning, verification,
  manifest, downsampled caching. CPU only, no GPU required.
- **PC (Phases 4–7)** — RTX 4060 Ti, 16 GB RAM. Frozen backbone feature extraction, probe training,
  robustness comparison.

Everything the laptop produces (`cache/` + `manifest.parquet`) is small enough to move to the PC on a
drive or via cloud storage — the multi-TB raw sequences never need to move.

## Setup

```bash
# 1. upstream tooling repo (gitignored here - it's third-party code)
git clone https://github.com/anthonytec2/OctoSense.git

# 2. environment (pip/venv rather than the upstream conda env - see note below)
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate  on Linux/macOS
pip install -r requirements-slim.txt
```

### Why `requirements-slim.txt` instead of upstream's `environment.yml`

Upstream ships a conda env that installs Python + FFmpeg + a CPU build of PyTorch/torchcodec, then all
of `requirements.txt`. For the laptop phases that's far more than we need, so this repo keeps a slim
pip-only set. Deliberately excluded:

| Excluded | Why |
|---|---|
| `torch`, `torchvision`, `torchcodec` | GPU inference — belongs to the PC phase (4+) |
| system FFmpeg | only needed by torchcodec; `opencv-python-headless` bundles its own decoder |
| `faiss-cpu`, `sentence-transformers`, `rank-bm25`, `fastapi` | semantic-search index — not used here |
| `ultralytics`, `transformers` | ground-truth *generation*; we consume the **shipped** depth/seg labels |
| `open3d`, `ouster-sdk`, `pupil-apriltags` | LiDAR/camera recalibration; we use the platform's existing calibration |
| `rerun-sdk` | visualization — added back in Phase 7 for the write-up clip |

## Project phases

| Phase | Machine | What | Status |
|---|---|---|---|
| 0 | Laptop | Environment setup | ✅ done |
| 1 | Laptop | Sequence selection → `sequences.txt` | ✅ done |
| 2 | Laptop | Download chosen sequences | ⬜ |
| 3 | Laptop | Cleaning, verification, manifest, caching | ⬜ |
| 4 | PC | Frozen feature extraction | ⬜ |
| 5 | PC | Probe training (depth / seg / steering) | ⬜ |
| 6 | PC | Robustness comparison | ⬜ |
| 7 | — | Write-up | ⬜ |

### Phase 1 — Sequence selection
Six **car** sequences: 2 daytime/clear, 2 night, 2 degraded. IDs and full rationale in
[`sequences.txt`](sequences.txt).

Day and night sequences are **route-matched** by GPS bounding-box overlap (IoU 0.98 and 0.94), so a
day→night performance drop reflects lighting rather than a change of scene.

#### Ground-truth availability constrains the experiment

Verified against the hub's actual file listing (`scripts/check_gt_files.py`), not assumed:

| Ground truth | Daytime | Night |
|---|---|---|
| `data.h5` (LiDAR/IMU/GPS/CAN) | 100% | 100% |
| depth | 100% | **100%** |
| segmentation | 100% | **0%** |
| RGB / stereo / thermal | 100% | 100% |

**Depth and steering support the full day→night robustness comparison. Segmentation does not** —
there is no night ground truth to score against, so the seg probe is a daytime-only,
in-distribution experiment. This is the main deviation from the original plan.

> Upstream's `download_octosense.py` annotates depth as "(car daytime)". **That comment is wrong** —
> depth is LiDAR-derived and present for all 68 night sequences.

### Phase 2 — Download
```bash
python OctoSense/download_octosense.py --platform car --no-events \
  --sequence <seq_id> --modalities data,depth,seg,rgb,ir --out ./octosense
```
`--no-events` skips the raw event stream, not needed for v1.

> **Note:** `--sequence` takes exactly **one** bag ID per call — there is no comma-list or repeated
> flag. Downloading N sequences means N invocations.

**Measured size** for the six chosen sequences (`scripts/estimate_size.py`): **~55.7 GB** without
events; events would add ~80 GB more (59% of full size). Dominated by `data.h5` (~28 GB) and depth
(~25 GB) — the RGB video is only ~1.5 GB total.

Recommended order: pull the `01_04` day/night matched pair **first** (~23 GB), validate the entire
Phase 3 pipeline on it, then fetch the remaining four. This avoids committing hours of download
before knowing the cleaning/manifest/cache code works.

### Phase 3 — Cleaning & verification
Integrity check → schema inventory (`docs/schema_notes.md`) → timestamp sync sanity check →
`manifest.parquet` (one row per synchronized frame) → downsampled JPEG `cache/` at 5 Hz →
basic stats and sanity plots.

**The split is by sequence, not by frame:** day sequences → train/val, night/degraded sequences →
held-out test. This is what makes the Phase 6 robustness comparison meaningful rather than
a memorization test.

### Phases 4–6 — Features, probes, robustness
Frozen encoders (DINOv2-small / ResNet18 / MobileNetV3 for RGB+thermal, a small CNN over projected
LiDAR range images, an MLP over IMU/CAN/GPS windows) run **once** over the cache. Then small
linear/shallow-MLP heads train on those features for three tasks — depth regression, segmentation,
and steering-angle regression. Each task runs three ways: **RGB-only, RGB+LiDAR, all modalities** —
then re-runs on the held-out night/degraded sequences.

## Repo layout

```
.
├── requirements-slim.txt   # laptop-phase dependencies
├── sequences.txt           # chosen sequence IDs + rationale (Phase 1)
├── docs/schema_notes.md    # HDF5 schema inventory (Phase 3)
├── scripts/
│   ├── list_repo_files.py       # locate metadata.parquet on the hub
│   ├── fetch_metadata.py        # download + summarise per-sequence metadata
│   ├── analyze_availability.py  # GT availability vs condition
│   ├── check_gt_files.py        # verify GT against real hub file listing
│   ├── select_sequences.py      # route-match day/night by GPS bbox IoU
│   └── estimate_size.py         # per-modality download size
├── OctoSense/              # gitignored - upstream tooling clone
├── octosense/              # gitignored - raw downloaded sequences (GB-TB)
└── cache/                  # gitignored - downsampled frames + manifest.parquet
```

## Open questions

- [x] Does `download_octosense.py` accept multiple `--sequence` values per call? — **No**, one per call.
- [x] Is there per-sequence condition metadata? — Yes, `metadata.parquet` at the repo root, 382 rows
      with `is_daytime`, `degraded`, `has_seg`, `gps_quality`, `sensor_dropout`, GPS bbox, and an
      official `train`/`test` split.
- [ ] Exact steering-angle / CAN field name (resolved in Phase 3 schema inventory).
- [ ] Which pretrained backbone fits comfortably in the 4060 Ti's VRAM for batched extraction.
- [ ] **Should night segmentation GT be generated?** Upstream ships `data_processing/seg_gt/` (EoMT).
      Running it on night sequences would restore seg to the robustness study, but it needs
      `transformers` + GPU and the labels would be model-generated on hard low-light input — of
      questionable quality as "ground truth". Deferred; depth carries the robustness story for now.

## Data quirks worth knowing

Gotchas found while reading `metadata.parquet` — they cost debugging time if unknown:

- `is_daytime`, `degraded`, `has_seg` are **object dtype holding Python bools**, so `~df["is_daytime"]`
  raises `KeyError` rather than negating. Coerce with `.astype(bool)` first.
- `sensor_dropout` uses the **literal string `"nan"`**, not a real null — `.notna()` matches every row.
  Filter with `.astype(str).ne("nan")`.
- `degraded` and `sensor_dropout` are **unrelated flags with zero overlap**: the 4 `degraded=True`
  sequences record no dropout, and all 5 sequences with real dropouts are `degraded=False`. Three of
  those 5 dropouts affect only the event stream, which this project doesn't download.

## License / citation

Upstream OctoSense is MIT licensed. If you build on the dataset, cite the paper:

```bibtex
@misc{bisulco2026octosense,
  title        = {{OctoSense}: Self-Supervised Learning for Multimodal Robot Perception},
  author       = {Bisulco, Anthony and Wang, Jeremy and Daniilidis, Kostas and Balestriero, Randall and Chaudhari, Pratik},
  year         = {2026},
  howpublished = {Preprint},
}
```
