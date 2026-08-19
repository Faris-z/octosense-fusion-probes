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
| 1 | Laptop | Sequence selection → `sequences.txt` | ⬜ |
| 2 | Laptop | Download chosen sequences | ⬜ |
| 3 | Laptop | Cleaning, verification, manifest, caching | ⬜ |
| 4 | PC | Frozen feature extraction | ⬜ |
| 5 | PC | Probe training (depth / seg / steering) | ⬜ |
| 6 | PC | Robustness comparison | ⬜ |
| 7 | — | Write-up | ⬜ |

### Phase 1 — Sequence selection
Pick ~4–6 short **car** sequences spanning 2 daytime/clear, 2 nighttime, and 1–2 explicitly
degraded-sensor recordings. IDs are recorded in `sequences.txt` so the selection is reproducible.

### Phase 2 — Download
```bash
python OctoSense/download_octosense.py --platform car --no-events \
  --sequence <seq_id> --modalities data,depth,seg,rgb,ir --out ./octosense
```
`--no-events` skips the raw event stream (~78% of a sequence's bytes), not needed for v1.

> **Note:** `--sequence` takes exactly **one** bag ID per call — there is no comma-list or repeated
> flag. Downloading N sequences means N invocations.

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
├── sequences.txt           # chosen sequence IDs (Phase 1)
├── docs/schema_notes.md    # HDF5 schema inventory (Phase 3)
├── scripts/                # verification / manifest / caching scripts
├── OctoSense/              # gitignored - upstream tooling clone
├── octosense/              # gitignored - raw downloaded sequences (GB-TB)
└── cache/                  # gitignored - downsampled frames + manifest.parquet
```

## Open questions

- [x] Does `download_octosense.py` accept multiple `--sequence` values per call? — **No**, one per call.
- [ ] Exact steering-angle / CAN field name (resolved in Phase 3 schema inventory).
- [ ] Which pretrained backbone fits comfortably in the 4060 Ti's VRAM for batched extraction.

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
