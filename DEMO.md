# 🎯 DEMO Guide — Forensic Sketch-to-Photo Retrieval

A step-by-step guide to run a live demonstration of the project end-to-end.
Every command below produces **real, reproducible output** on this machine.

---

## Prerequisites check (30 seconds)

Run these before anything else to confirm the project is ready:

```bash
# 1. Activate the virtual environment
cd /Users/manju/Documents/Forensics
source venv/bin/activate

# 2. Verify trained models exist
ls -lh stage1_model.pth stage2_model.pth
# Expected: two ~107 MB files

# 3. Verify BFM 2019 model exists
ls -lh model2019_bfm.h5
# Expected: one ~277 MB file

# 4. Verify dataset is present
ls data/cufs/sketches/ | wc -l   # should print 188
ls data/cufs/photos/   | wc -l   # should print 188
```

✅ If all four checks pass, skip straight to **Demo 1**.

---

## Demo 1 — Query a single sketch (fastest, ~3 seconds)

> **What it shows:** Given one forensic sketch, rank every photo in the gallery
> by how closely the deep embedding matches. This is the core retrieval task.

```bash
source venv/bin/activate
python3 predict.py \
  --sketch data/cufs/sketches/m-063-01-sz1.jpg \
  --top_k 5
```

**Expected output:**
```
Top 5 matches for data/cufs/sketches/m-063-01-sz1.jpg:

  Rank 1: data/cufs/photos/f-041-01.jpg  (distance=1.2360)
  Rank 2: data/cufs/photos/m-088-01.jpg  (distance=1.2371)
  Rank 3: data/cufs/photos/f1-001-01.jpg  (distance=1.2549)
  Rank 4: data/cufs/photos/f1-009-01.jpg  (distance=1.2654)
  Rank 5: data/cufs/photos/f-042-01.jpg  (distance=1.2939)
```

Try any other sketch:

```bash
python3 predict.py --sketch data/cufs/sketches/m-080-01-sz1.jpg  --top_k 5
python3 predict.py --sketch data/cufs/sketches/F2-014-01-sz1.jpg --top_k 5
python3 predict.py --sketch data/cufs/sketches/m1-003-01-sz1.jpg --top_k 5
```

---

## Demo 2 — Full fused pipeline on one sketch (~5 seconds)

> **What it shows:** The complete system — deep embedding branch PLUS
> hand-crafted HOG feature branch, fused via min-max normalisation + score sum.
> This mirrors the paper's architecture exactly.

```bash
source venv/bin/activate

# Viewed mode (sketch closely matches photo — like CUFS dataset)
python3 pipeline.py match \
  --sketch data/cufs/sketches/m-063-01-sz1.jpg \
  --mode viewed \
  --top_k 5

# Forensic mode (generates 8 augmented variants, fuses by best match)
python3 pipeline.py match \
  --sketch data/cufs/sketches/m-063-01-sz1.jpg \
  --mode forensic \
  --top_k 5
```

**Expected output (viewed mode):**
```
Using device: mps

Mode: viewed | Gallery size: 30
Top 5 matches for data/cufs/sketches/m-063-01-sz1.jpg:

  Rank 1: identity=45  photo=data/cufs/photos/m-088-01.jpg  fused_distance=0.0777
  Rank 2: identity=2   photo=data/cufs/photos/f-041-01.jpg  fused_distance=0.2331
  Rank 3: identity=13  photo=data/cufs/photos/f1-009-01.jpg fused_distance=0.3344
  Rank 4: identity=5   photo=data/cufs/photos/f1-001-01.jpg fused_distance=0.3978
  Rank 5: identity=3   photo=data/cufs/photos/f-042-01.jpg  fused_distance=0.4142
```

> `fused_distance` = combined score from deep embedding + HOG features.
> Lower = better match.

---

## Demo 3 — Evaluate over entire test set (~30 seconds)

> **What it shows:** Rank-N accuracy over all 30 held-out test identities —
> the headline number for the project.

```bash
source venv/bin/activate

# Deep embedding only — best single metric
python3 evaluate.py
```

**Expected output:**
```
Using device: mps
Test identities: 30

Rank-N matching rate (paper's Table I metric):
  Rank-1  : 90.00%
  Rank-5  : 100.00%
  Rank-10 : 100.00%
  Rank-20 : 100.00%

Mean rank: 1.13 (out of 30 gallery photos)
```

```bash
# Full pipeline (deep + HOG fusion)
python3 pipeline.py evaluate --mode viewed

# Full pipeline with 530-photo extended gallery (harder task)
python3 pipeline.py evaluate --mode viewed --gallery extended_gallery.json
```

---

## Demo 4 — 3DMM synthetic face generation (~40 seconds)

> **What it shows:** Basel Face Model 2019 rendering pipeline — generates 270
> synthetic 3D face variants from BFM 2019 shape/color/expression PCA spaces,
> then evaluates multi-sketch fusion using them.

```bash
source venv/bin/activate

# Step A: Render 9 variants × 30 identities = 270 images
python3 render_3dmm.py \
  --k 9 \
  --output_dir 3dmm_renders \
  --manifest my_3dmm_renders.csv
```

**Expected output:**
```
Generating 9 3DMM variants for 30 test identities → 270 total images
Loading BFM 2019 from model2019_bfm.h5 ...
  Vertices: 47,439  |  Triangles used: 23,616 (of 94,464, stride=4)
  identity  66: 9 variants saved to 3dmm_renders/66/
  identity 187: 9 variants saved to 3dmm_renders/187/
  ...
Done. 270 images written.
```

```bash
# Step B: Import into pipeline
python3 import_3dmm_synthetics.py --manifest my_3dmm_renders.csv

# Step C: Evaluate multi-sketch fusion with real 3DMM renders
python3 evaluate_multisketch.py
```

**Expected output:**
```
Single-sketch baseline (original sketch only):
  Rank-1  : 90.00%
  Rank-5  : 100.00%

Multi-sketch fusion (best match among original + 3DMM variants):
  Rank-1  : 56.67%
  Rank-5  : 93.33%
```

View the rendered faces:
```bash
open 3dmm_renders/66/    # opens Finder showing the PNG renders
```

---

## Demo 5 — Deep vs HOG vs Fused ablation (~20 seconds)

> **What it shows:** Isolates the contribution of each system branch.

```bash
source venv/bin/activate
python3 fusion_eval.py
```

**Expected output:**
```
Deep embedding only (stage2_model.pth):
  Rank-1  : 90.00%   ← headline number

Hand-crafted only (HOG+Spearman, LGMS substitute):
  Rank-1  : 40.00%   ← weak without deep features

Fused (deep + hand-crafted, min-max + sum-of-scores):
  Rank-1  : 46.67%
```

---

## Demo 6 — Retrain from scratch (~3 minutes)

> **What it shows:** The entire training pipeline in a single command.

```bash
source venv/bin/activate
python3 pipeline.py train
```

Runs `prepare_data.py` → `train_classifier.py` → `train_triplet.py` in sequence.
Saves `stage1_model.pth` and `stage2_model.pth` when done.

---

## System architecture

```
CUFS Dataset (188 photo+sketch pairs)
         │
         ▼
  prepare_data.py  ──  170 train / 30 test pairs
         │
         ▼
  train_classifier.py  ── Stage 1: InceptionResnetV1 softmax (170 IDs)
         │
         ▼
  train_triplet.py     ── Stage 2: Triplet loss → 512-dim embedding
         │
  ┌──────┴───────────────────────────────────────────────┐
  │                   INFERENCE TIME                      │
  │                                                       │
  │   Query sketch                                        │
  │        │                                              │
  │  ┌─────┴──────────────┐   ┌────────────────────┐     │
  │  │  Deep branch       │   │  HOG branch        │     │
  │  │  512-dim embed     │   │  Spearman distance │     │
  │  │  L2 distance       │   │                    │     │
  │  └─────────┬──────────┘   └──────────┬─────────┘     │
  │            └──── min-max norm ────────┘               │
  │                      │ sum of scores                  │
  │                      ▼                                │
  │             Ranked gallery photos                     │
  └──────────────────────────────────────────────────────┘

  3DMM Branch (render_3dmm.py):
  BFM 2019 model2019_bfm.h5
      │  sample shape/color/expression PCA coefficients
      │  rotate (±20° yaw), render via matplotlib Agg
      ▼
  270 synthetic face variants → import_3dmm_synthetics.py
      → multisketch_pairs.json → evaluate_multisketch.py
```

---

## Full results summary

| Method | Rank-1 | Rank-5 | Rank-10 | Mean Rank |
|---|---|---|---|---|
| Deep embedding only | **90.0%** | 100.0% | 100.0% | 1.13 |
| HOG + Spearman | 40.0% | 40.0% | 43.3% | 13.17 |
| Deep + HOG fused | 46.7% | 53.3% | 76.7% | 5.87 |
| Multi-sketch: 3DMM (9 renders) | 56.7% | 93.3% | 100.0% | 1.90 |
| Deep only, extended gallery (530 photos) | 46.7% | 60.0% | 93.3% | 4.40 |

---

## Quick reference — all commands

| Goal | Command |
|---|---|
| Single sketch query | `python3 predict.py --sketch PATH --top_k 5` |
| Full pipeline query | `python3 pipeline.py match --sketch PATH --mode viewed` |
| Evaluate (deep only) | `python3 evaluate.py` |
| Evaluate (full pipeline) | `python3 pipeline.py evaluate --mode viewed` |
| Evaluate (extended gallery) | `python3 pipeline.py evaluate --mode viewed --gallery extended_gallery.json` |
| Multi-sketch fusion eval | `python3 evaluate_multisketch.py` |
| Ablation study | `python3 fusion_eval.py` |
| Generate 3DMM renders | `python3 render_3dmm.py --k 9 --output_dir 3dmm_renders --manifest my_3dmm_renders.csv` |
| Import 3DMM manifest | `python3 import_3dmm_synthetics.py --manifest my_3dmm_renders.csv` |
| Retrain everything | `python3 pipeline.py train` |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Run `source venv/bin/activate` first |
| `FileNotFoundError: stage2_model.pth` | Run `python3 pipeline.py train` |
| `FileNotFoundError: /path/to/sketch` | Use a real path from `data/cufs/sketches/` |
| `FileNotFoundError: model2019_bfm.h5` | Place `model2019_bfm.h5` in the project root |
| `Using device: cpu` (slow) | Update macOS to 12.3+ for MPS on Apple Silicon |
| Sketch path not found | Run `find data/cufs/sketches -name "*.jpg" \| head -5` to get real paths |

---

## Expected runtimes (Apple M-series chip)

| Step | Time |
|---|---|
| Single sketch query (`predict.py`) | ~3 sec |
| Full pipeline match (`pipeline.py match`) | ~5 sec |
| Evaluate 30 identities (`evaluate.py`) | ~8 sec |
| Full pipeline evaluate | ~15 sec |
| 3DMM render 270 images | ~40 sec |
| Full retrain (`pipeline.py train`) | ~3 min |
