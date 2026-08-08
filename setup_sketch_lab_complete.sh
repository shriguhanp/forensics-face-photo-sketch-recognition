#!/bin/bash
# Creates the ENTIRE sketch-lab project from scratch: 18 files covering
# data prep, training, the paper-gap experiments (multi-sketch fusion,
# LGMS-substitute fusion, extended gallery), and the unified end-to-end
# pipeline. Run this in an EMPTY folder you want to become your project:
#   mkdir -p ~/sketch-lab && cd ~/sketch-lab && bash setup_sketch_lab_complete.sh
set -e

cat > 'README.md' << 'SKETCHLAB_EOF'
# Practical Lab: Face Photo-Sketch Recognition (Deep Transfer Learning)

This implements the **core methodology** of Galea & Farrugia, *"Forensic Face
Photo-Sketch Recognition Using a Deep Learning-Based Architecture"* (IEEE
TIFS, 2025): given a hand-drawn sketch of a face, retrieve the matching photo
from a gallery, using transfer learning on a pretrained face-recognition
network, tuned in two stages, evaluated with the paper's Rank-N metric.

**This version has a checkpoint after every step.** Do not move to the next
step until the current one's checkpoint passes. This is the single biggest
thing that will save you time.

## Honest scope: what's faithful vs. what's substituted

Three things in the original paper aren't practically reproducible outside a
research lab, so this substitutes the closest open alternative:

1. **Backbone.** Paper uses VGG-Face (not freely redistributable). We use
   **InceptionResnetV1 pretrained on VGGFace2** via `facenet-pytorch`.
2. **Augmentation.** Paper generates synthetic training images with a
   licensed **3D Morphable Model**. We use standard 2D augmentation (flips,
   rotation, color jitter) instead.
3. **Dataset.** Paper also evaluates on a restricted real forensic sketch
   set (not public). We use only the public **CUFS** dataset (606
   photo/sketch pairs) via Kaggle.

Given these substitutions, expect noticeably lower accuracy than the paper's
published numbers. That's expected — the goal here is the working pipeline,
not matching a benchmark.

---

## Project layout

```
sketch-lab/
├── README.md
├── requirements.txt
├── download_data.sh
├── pairing.py
├── explore_data.py
├── prepare_data.py
├── model_utils.py
├── train_classifier.py
├── train_triplet.py
├── evaluate.py
├── predict.py
├── generate_synthetic_sketches.py   (Part 2 — multi-sketch fusion)
├── evaluate_multisketch.py           (Part 2)
├── handcrafted_features.py           (Part 2 — LGMS-substitute fusion)
├── fusion_eval.py                     (Part 2)
├── extend_gallery.py                  (Part 2 — extended gallery)
├── evaluate_extended_gallery.py       (Part 2)
└── pipeline.py                         (Part 3 — unified end-to-end entry point)
```

Put all these files in one folder. Every command below runs in the VS Code
integrated terminal, from inside that folder.

---

## STEP 1 — Create the project and virtual environment

```bash
mkdir -p ~/sketch-lab && cd ~/sketch-lab
python3 -m venv venv
source venv/bin/activate
```

**✅ Checkpoint:** your terminal prompt now starts with `(venv)`. If it
doesn't, stop — run `source venv/bin/activate` again before continuing.

Copy all 11 files listed above into this `~/sketch-lab` folder now.

In VS Code: File → Open Folder → select `~/sketch-lab`. Then ⇧⌘P →
"Python: Select Interpreter" → pick the one inside `venv/bin/python`.

---

## STEP 2 — Install packages one at a time

Installing all at once hides which package failed if something goes wrong.
Run these one line at a time, **letting each finish before starting the
next**:

```bash
pip install --upgrade pip
pip install torch
pip install torchvision
pip install facenet-pytorch
pip install matplotlib
pip install pillow
pip install numpy
pip install scikit-learn
pip install kaggle
```

**✅ Checkpoint — run this exact command:**
```bash
python3 -c "import torch, torchvision, facenet_pytorch, matplotlib, PIL, numpy, sklearn, kaggle; print('ALL OK')"
```
You must see `ALL OK` printed with no errors before continuing. If you get
`ModuleNotFoundError: No module named 'X'`, run `pip install X` again and
re-check.

**✅ Checkpoint — GPU check:**
```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
You should see `MPS available: True`.

---

## STEP 3 — Get a Kaggle API key

CUFS is hosted on Kaggle, which requires a free account and an API key.
**Do this manually rather than relying on the browser download** — it's
more reliable.

1. Create a free account at https://www.kaggle.com if you don't have one.
2. Go to https://www.kaggle.com/settings
3. Scroll to the **API** section → click **Create New Token**.
4. A popup or small download will show your username and key. If a file
   downloads, note where — check with:
   ```bash
   find ~ -iname "kaggle.json" 2>/dev/null
   ```
5. Create the credentials file yourself (this works regardless of whether
   the download worked) — **type this directly in your own terminal, not
   anywhere else**, since it contains a secret key:
   ```bash
   mkdir -p ~/.kaggle
   cat > ~/.kaggle/kaggle.json << 'EOF'
   {"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_API_KEY"}
   EOF
   chmod 600 ~/.kaggle/kaggle.json
   ```
   Replace `YOUR_KAGGLE_USERNAME` and `YOUR_API_KEY` with your actual
   values before running.

**⚠️ Never paste your actual key into a chat, ticket, or anywhere outside
your own terminal.** If you ever do accidentally expose one, go back to
Kaggle settings and click "Create New Token" again — this invalidates the
old one.

**✅ Checkpoint:**
```bash
cat ~/.kaggle/kaggle.json
```
You should see `{"username":"...","key":"..."}` with real values, not
placeholders.

---

## STEP 4 — Download the dataset

```bash
chmod +x download_data.sh
./download_data.sh
```

**✅ Checkpoint:** the script should end by printing a directory listing
under `data/cufs`. If instead you get a 401/403 error, your `kaggle.json`
either isn't in `~/.kaggle/` or has the wrong permissions — redo Step 3's
checkpoint. If you get "dataset already present," delete `data/` and rerun
if you want a fresh download.

---

## STEP 5 — Inspect the data before training anything

```bash
python3 explore_data.py
```

This prints the folder structure it found and how many images are in each,
and saves `sample_pairs.png`.

**✅ Checkpoint:** open `sample_pairs.png` in VS Code (click it in the file
explorer). Each column must show the SAME person's photo and sketch. If
they don't match, stop here and tell me exactly what `explore_data.py`
printed for the folder structure — don't proceed to training with
mismatched pairs.

---

## STEP 6 — Build identity pairs and train/test split

```bash
python3 prepare_data.py
```

**✅ Checkpoint:** you should now have `train_pairs.json` and
`test_pairs.json` in the folder, and the script prints a total identity
count roughly matching what Step 5 reported (around 600 for full CUFS).

---

## STEP 7 — Stage 1: classification fine-tuning

```bash
python3 train_classifier.py
```

Trains the pretrained network to treat each identity's photo and sketch as
the same class. Takes a few minutes on an M4.

**✅ Checkpoint:** training accuracy should climb epoch over epoch (printed
each epoch) and the script ends by printing `Saved stage1_model.pth`.
Confirm the file exists:
```bash
ls -la stage1_model.pth
```

---

## STEP 8 — Stage 2: triplet embedding fine-tuning

```bash
python3 train_triplet.py
```

Fine-tunes further so embedding distance reflects same/different identity.

**✅ Checkpoint:** triplet loss should generally trend downward across
epochs, and you should see `Saved stage2_model.pth`:
```bash
ls -la stage2_model.pth
```

---

## STEP 9 — Evaluate

```bash
python3 evaluate.py
```

**✅ Checkpoint:** prints Rank-1/5/10/20 matching rates on held-out test
identities. Any output here (even low percentages) means the pipeline
worked end to end — this is expected to be lower than the paper's numbers
for the reasons in the "Honest scope" section above.

---

## STEP 10 — Use it: query with a sketch

```bash
python3 predict.py --sketch /path/to/a/test/sketch.jpg --top_k 5
```

Tip: grab a real sketch path from `test_pairs.json` to try this on a known
example first, before trying an arbitrary new sketch.

**✅ Checkpoint:** prints a ranked list of candidate photo paths with
distances — this is the actual sketch-to-photo retrieval the paper targets.

---

## PART 2 — Closing three more gaps from the paper

Steps 1–10 above cover the core pipeline. This section implements three more
pieces from the paper that weren't in the original scope: multi-sketch
test-time fusion (DEEPS-M), fusion with a hand-crafted method (LGMS
substitute), and an extended/harder gallery. **A fourth gap — the paper's
3D Morphable Model synthesis — genuinely cannot be code-generated; see the
note at the very end of this section for what that would actually require.**

New files this section uses:
```
generate_synthetic_sketches.py
evaluate_multisketch.py
handcrafted_features.py
fusion_eval.py
extend_gallery.py
evaluate_extended_gallery.py
```

Run these after you've completed Steps 1–9 (you need `stage2_model.pth`
and `test_pairs.json` to already exist).

### STEP 11 — Multi-sketch test-time fusion (DEEPS-M substitute)

The paper doesn't just compare one sketch per subject at test time — it
generates ~200 synthetic variants per sketch (via the 3DMM) and fuses the
distances. We can't generate 3DMM variants, but we can test the *fusion
mechanism itself* using augmentation-based variants instead:

```bash
python3 generate_synthetic_sketches.py --k 8
python3 evaluate_multisketch.py
```

**✅ Checkpoint:** prints Rank-N for the single-sketch baseline AND the
multi-sketch fused result side by side, plus a per-identity breakdown of
how many identities improved vs. worsened with fusion. **Don't be surprised
if this doesn't clearly beat the baseline** — 2D augmentation just perturbs
pixels, it can't correct facial-attribute distortions the way a 3DMM does.
A null or mixed result here is a legitimate, reportable finding, not a bug.

### STEP 12 — Fusion with a hand-crafted method (LGMS substitute)

The paper's best numbers come from fusing DEEPS with LGMS (log-Gabor +
MLBP + Spearman correlation — a separate paper's method). We substitute a
HOG+Spearman hand-crafted baseline, fused the same way (min-max
normalization + sum-of-scores):

```bash
python3 fusion_eval.py
```

**✅ Checkpoint:** prints three Rank-N tables — deep-only, hand-crafted-only,
and fused. Expect the hand-crafted-only numbers to be noticeably weaker
than the paper's LGMS (HOG is a much simpler descriptor); the interesting
result is whether fusion still helps even with a weaker second method.

### STEP 13 — Extended gallery (harder, more realistic retrieval)

The paper pads its test gallery with thousands of extra photos from
restricted/licensed datasets to simulate a real mugshot database. We use
LFW (freely available) as distractor photos instead:

```bash
python3 extend_gallery.py --n_distractors 500
python3 evaluate_extended_gallery.py
```

**✅ Checkpoint:** prints the gallery composition (real photos + distractors)
and Rank-N against this larger gallery. Compare against Step 9's numbers —
expect Rank-N to drop, since there are now more distractor photos to
confuse the model. **That drop is the point** — it demonstrates how much
small-gallery evaluations (including Step 9's) can overstate real-world
performance.

If `extend_gallery.py` fails to download LFW (no internet access in your
environment), manually download
`http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz`, extract it, and
rerun with `--lfw_dir /path/to/extracted/lfw-deepfunneled`.

### The one gap that's not code-generatable: 3D Morphable Model synthesis

This is the paper's actual headline contribution, and it genuinely requires
manual setup, not just more scripts:

1. Register for the **Basel Face Model** (free for non-commercial/research
   use, requires filling out a license form): http://faces.cs.unibas.ch/bfm/main.php
2. Get the model-fitting code: https://github.com/waps101/3DMM_edges
   (fits the 3DMM to a 2D photo/sketch by matching edges — this is a
   research codebase, expect to spend real time getting it running, likely
   MATLAB-based)
3. Fit the model to each CUFS photo and sketch, then vary the fitted
   parameters (facial features + global attributes: weight, age, height,
   gender) to render synthetic images — the paper's supplementary material
   (http://wp.me/P6CDe8-7D) has their exact parameter ranges
4. Once you have synthetic images, they slot into this project exactly
   where `generate_synthetic_sketches.py` currently uses augmentation —
   replace that script's output with your 3DMM renders and rerun Steps 7–13

This is realistically a multi-week project on its own, not a "few days on
Colab" addition — flagging it honestly rather than pretending otherwise.

---

## PART 3 — Running it as ONE pipeline, not separate scripts

Steps 1–13 above run each piece in isolation, which is good for
understanding and ablation, but doesn't match how the paper's system
actually works: one continuous flow where every sketch goes through
training, then a branch depending on sketch type, then fusion with a
second method, ending in a single ranked match list.

`pipeline.py` implements that full flow as one entry point:

```
Photo+sketch pair -> [augmentation, substituting 3DMM] -> fine-tune backbone
-> trained embedding model
    -> "viewed" sketches: use the model directly
    -> "forensic" sketches: generate K variants, fuse by best-match
-> [merge] -> fuse with hand-crafted method (LGMS substitute)
-> ranked match list
```

You still need Steps 1–6 done first (venv, packages, dataset, pairing).
From there, three commands replace almost everything else:

### Train everything in one command

```bash
python3 pipeline.py train
```

Runs `prepare_data.py` → `train_classifier.py` → `train_triplet.py` in
sequence, stopping immediately if any step fails. Equivalent to Steps 6–8,
just not needing to run them one at a time.

**✅ Checkpoint:** `stage1_model.pth` and `stage2_model.pth` both exist.

### Query one sketch through the full pipeline

```bash
python3 pipeline.py match --sketch /path/to/sketch.jpg --mode viewed --top_k 5
```

For a sketch that closely resembles its photo (like CUFS sketches), use
`--mode viewed`. For a sketch with real distortions (more like a real
forensic sketch), use `--mode forensic` — this triggers the multi-sketch
generation-and-fusion branch instead of a single direct comparison.

**✅ Checkpoint:** prints a ranked list of gallery matches with fused
distances — this single number already combines the deep embedding AND the
hand-crafted method, exactly like the diagram's final box.

### Evaluate the full pipeline over the whole test set

```bash
python3 pipeline.py evaluate --mode viewed
python3 pipeline.py evaluate --mode forensic
python3 pipeline.py evaluate --mode viewed --gallery extended_gallery.json
```

The last variant needs `extend_gallery.py` run first (Step 13) — it plugs
the extended gallery directly into the same unified pipeline instead of
needing a separate script.

**✅ Checkpoint:** prints Rank-N for the WHOLE pipeline working together —
this is the number that actually reflects what the diagram describes, not
any single piece of it in isolation.

### How this relates to the individual scripts from Parts 1–2

`evaluate.py`, `evaluate_multisketch.py`, `fusion_eval.py`, and
`evaluate_extended_gallery.py` still work standalone — keep using them when
you want to isolate one variable at a time (e.g., "does fusion alone help,
holding the gallery size constant?"). `pipeline.py` is what you run when
you want the answer to "how does the whole system perform," and it's what
you should report as your headline number if you publish this.

---

## Troubleshooting index

- **`ModuleNotFoundError`** → Step 2 checkpoint wasn't fully passed; go back
  and run `pip install <missing package>`.
- **`kaggle: 401/403`** → Step 3 checkpoint; check `~/.kaggle/kaggle.json`
  contents and permissions (`chmod 600`).
- **Pairing looks wrong in `sample_pairs.png`** → open `pairing.py`, check
  the `PAIRING STRATEGY` comment near the top, adjust `PHOTO_KEYWORDS` /
  `SKETCH_KEYWORDS` / `SKETCH_SUFFIX_PATTERN` to match what Step 5 printed.
- **MPS available: False** → training still works, just slower on CPU;
  update macOS if you expect MPS support on an M4.
- **Anything else** → paste the exact command you ran and its full output,
  not just a description of what happened.
SKETCHLAB_EOF

cat > 'requirements.txt' << 'SKETCHLAB_EOF'
torch
torchvision
facenet-pytorch
matplotlib
pillow
numpy
scikit-learn
scikit-image
scipy
kaggle
SKETCHLAB_EOF

cat > 'download_data.sh' << 'SKETCHLAB_EOF'
#!/bin/bash
# Downloads the CUHK Face Sketch Database (CUFS) from Kaggle.
# Requires a Kaggle API token at ~/.kaggle/kaggle.json (see README section 2).

set -e

mkdir -p data
cd data

if [ -d "cufs" ] && [ "$(ls -A cufs 2>/dev/null)" ]; then
  echo "Dataset already present at data/cufs — skipping download."
  exit 0
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found."
  echo "Get an API token from https://www.kaggle.com/settings -> API -> Create New Token"
  echo "then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

echo "Downloading CUFS from Kaggle..."
kaggle datasets download -d arbazkhan971/cuhk-face-sketch-database-cufs -p cufs --unzip

echo "Done. Contents of data/cufs:"
find cufs -maxdepth 3 -type d
echo ""
echo "Next: run 'python3 explore_data.py' to inspect the actual structure."
SKETCHLAB_EOF

cat > 'pairing.py' << 'SKETCHLAB_EOF'
"""
pairing.py
Shared logic for finding (photo, sketch) pairs inside the CUFS dataset
folder, whatever its exact internal layout turns out to be (Kaggle mirrors
of research datasets are not always organized consistently).

PAIRING STRATEGY
-----------------
1. Walk the dataset root and classify every image as "photo" or "sketch"
   based on keywords in its path (folder name, then filename as a
   fallback).
2. Group images by their containing "subset" directory (e.g. CUHK / AR /
   XM2VTS each have their own photo+sketch subfolders), so we never
   accidentally pair a photo from one subset with a sketch from another.
3. Within each subset, try to match a photo to its sketch by stripping
   common sketch-only suffixes/prefixes ("-sz1", "_sketch", etc.) from the
   filename and matching the remaining key.
4. If that yields no matches (unfamiliar naming convention) but the photo
   and sketch counts are equal, fall back to pairing by sorted order —
   true for CUFS-style datasets where sorting preserves subject order.

If pairing looks wrong for your download, adjust SKETCH_SUFFIX_PATTERN or
PHOTO_KEYWORDS / SKETCH_KEYWORDS below based on what explore_data.py prints.
"""

import os
import re

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PHOTO_KEYWORDS = ("photo", "photos", "images")
SKETCH_KEYWORDS = ("sketch", "sketches", "drawing")
SKETCH_SUFFIX_PATTERN = re.compile(r"(-sz\d+|_sketch|-sketch|sketch)$", re.IGNORECASE)


def classify_image(path):
    """Return 'photo' or 'sketch' based on path keywords, or None if unclear."""
    lower_path = path.lower()
    for kw in SKETCH_KEYWORDS:
        if kw in lower_path:
            return "sketch"
    for kw in PHOTO_KEYWORDS:
        if kw in lower_path:
            return "photo"
    return None


def find_all_images(root):
    """Return list of (full_path, subset_dir) for every image under root."""
    images = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(dirpath, fname))
    return images


def base_key(filename):
    stem = os.path.splitext(filename)[0]
    stem = SKETCH_SUFFIX_PATTERN.sub("", stem)
    return stem.lower()


def find_pairs(root, verbose=True):
    """
    Returns a list of (photo_path, sketch_path) tuples found under root.
    """
    all_images = find_all_images(root)

    photos, sketches = [], []
    unclassified = []
    for path in all_images:
        cls = classify_image(path)
        if cls == "photo":
            photos.append(path)
        elif cls == "sketch":
            sketches.append(path)
        else:
            unclassified.append(path)

    if verbose:
        print(f"Found {len(photos)} photo-classified, {len(sketches)} "
              f"sketch-classified, {len(unclassified)} unclassified images.")

    # Group by immediate subset directory: two levels above photo/sketch dir
    def subset_of(path):
        # e.g. .../cufs/CUHK/photos/1.jpg -> .../cufs/CUHK
        return os.path.dirname(os.path.dirname(path))

    subsets = sorted(set(subset_of(p) for p in photos) | set(subset_of(s) for s in sketches))

    pairs = []
    for subset in subsets:
        subset_photos = [p for p in photos if subset_of(p) == subset]
        subset_sketches = [s for s in sketches if subset_of(s) == subset]
        if not subset_photos or not subset_sketches:
            continue

        # Try key-based matching first
        photo_by_key = {base_key(os.path.basename(p)): p for p in subset_photos}
        sketch_by_key = {base_key(os.path.basename(s)): s for s in subset_sketches}
        common_keys = set(photo_by_key) & set(sketch_by_key)

        if len(common_keys) >= min(len(subset_photos), len(subset_sketches)) * 0.5:
            for key in sorted(common_keys):
                pairs.append((photo_by_key[key], sketch_by_key[key]))
            if verbose:
                print(f"  {subset}: matched {len(common_keys)} pairs by filename key.")
        elif len(subset_photos) == len(subset_sketches):
            # Fallback: pair by sorted order
            for p, s in zip(sorted(subset_photos), sorted(subset_sketches)):
                pairs.append((p, s))
            if verbose:
                print(f"  {subset}: filename matching failed, paired "
                      f"{len(subset_photos)} by sorted order (verify with sample_pairs.png).")
        else:
            if verbose:
                print(f"  {subset}: could not pair — {len(subset_photos)} photos vs "
                      f"{len(subset_sketches)} sketches, and no filename-key overlap.")

    return pairs
SKETCHLAB_EOF

cat > 'explore_data.py' << 'SKETCHLAB_EOF'
"""
explore_data.py
Inspect the raw CUFS download and preview the auto-detected photo/sketch
pairing before you trust it for training.

Run: python3 explore_data.py
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

from pairing import find_pairs

DATA_ROOT = "data/cufs"


def print_tree_summary(root):
    print(f"Directory structure under {root}:\n")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        if depth > 3:
            continue
        images = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        indent = "  " * depth
        label = os.path.basename(dirpath) or dirpath
        if images:
            print(f"{indent}{label}/  ({len(images)} images)")
        else:
            print(f"{indent}{label}/")


def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"Could not find {DATA_ROOT}. Run ./download_data.sh first.")
        return

    print_tree_summary(DATA_ROOT)
    print("\nAttempting to auto-pair photos with sketches...\n")
    pairs = find_pairs(DATA_ROOT, verbose=True)
    print(f"\nTotal pairs found: {len(pairs)}")

    if not pairs:
        print("\nNo pairs found. Open pairing.py and adjust PHOTO_KEYWORDS / "
              "SKETCH_KEYWORDS / SKETCH_SUFFIX_PATTERN to match your actual "
              "folder and filename structure shown above.")
        return

    # Preview a handful of pairs so you can visually confirm correctness
    sample = random.sample(pairs, min(6, len(pairs)))
    fig, axes = plt.subplots(2, len(sample), figsize=(3 * len(sample), 6))
    for col, (photo_path, sketch_path) in enumerate(sample):
        axes[0, col].imshow(Image.open(photo_path).convert("RGB"))
        axes[0, col].set_title("photo", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(Image.open(sketch_path).convert("RGB"))
        axes[1, col].set_title("sketch", fontsize=9)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig("sample_pairs.png", dpi=120)
    print("\nSaved sample_pairs.png — open it and confirm each column shows "
          "the SAME person's photo and sketch before proceeding to prepare_data.py.")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'prepare_data.py' << 'SKETCHLAB_EOF'
"""
prepare_data.py
Builds the identity list from matched (photo, sketch) pairs and splits
identities into train/test sets — mirroring the paper's protocol of testing
on subjects never seen during training (Section III-D).

Run: python3 prepare_data.py
Produces: train_pairs.json, test_pairs.json
"""

import json
import random

from pairing import find_pairs

DATA_ROOT = "data/cufs"
TEST_FRACTION = 0.15
SEED = 42


def main():
    random.seed(SEED)

    pairs = find_pairs(DATA_ROOT, verbose=True)
    if not pairs:
        print("No pairs found — run explore_data.py first and fix pairing.py if needed.")
        return

    # Each (photo, sketch) pair IS one identity/class (CUFS has exactly one
    # photo and one sketch per subject).
    identities = [
        {"identity_id": i, "photo": photo, "sketch": sketch}
        for i, (photo, sketch) in enumerate(pairs)
    ]
    random.shuffle(identities)

    n_test = max(1, int(len(identities) * TEST_FRACTION))
    test_identities = identities[:n_test]
    train_identities = identities[n_test:]

    with open("train_pairs.json", "w") as f:
        json.dump(train_identities, f, indent=2)
    with open("test_pairs.json", "w") as f:
        json.dump(test_identities, f, indent=2)

    print(f"Total identities: {len(identities)}")
    print(f"Train identities: {len(train_identities)} -> train_pairs.json")
    print(f"Test identities:  {len(test_identities)} -> test_pairs.json")
    print("\nNo identity appears in both files, matching the paper's protocol "
          "of evaluating on subjects unseen during training.")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'model_utils.py' << 'SKETCHLAB_EOF'
"""
model_utils.py
Shared preprocessing and backbone-loading code used by train_classifier.py,
train_triplet.py, evaluate.py, and predict.py, so all stages use identical
image preprocessing (critical for embeddings to be comparable).
"""

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from facenet_pytorch import InceptionResnetV1

IMAGE_SIZE = 160  # InceptionResnetV1's expected input size
EMBEDDING_DIM = 512


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_image_tensor(path, augment=False):
    """
    Loads an image and preprocesses it exactly the way InceptionResnetV1
    (pretrained on VGGFace2) expects: RGB, resized to 160x160, standardized
    to roughly [-1, 1] via (pixel - 127.5) / 128.
    Works for both photos and (often grayscale) sketches.

    If augment=True, applies light random augmentation (flip, small rotation,
    brightness/contrast jitter) — our stand-in for the paper's 3D Morphable
    Model synthetic image generation (see README section 9). Only use this
    for training data, never for evaluation/prediction.
    """
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    if augment:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, fillcolor=(128, 128, 128))
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    arr = np.array(img).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    tensor = (tensor - 127.5) / 128.0
    return tensor


def build_backbone(device, freeze_until_block=6):
    """
    Loads InceptionResnetV1 pretrained on VGGFace2 and freezes its early
    layers, leaving later blocks trainable — the same "don't relearn edges
    and eyes from scratch" logic as the paper's use of a pretrained VGG-Face.

    freeze_until_block: how many of the early repeated blocks to freeze.
    InceptionResnetV1 exposes named blocks (conv2d_1a ... block8, mixed_7a,
    repeat_3, etc.); we freeze everything up to and including `repeat_1`
    by default and leave the rest trainable.
    """
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)

    trainable_from = ["repeat_2", "mixed_7a", "repeat_3", "block8", "last_linear", "last_bn"]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable_from)

    return model
SKETCHLAB_EOF

cat > 'train_classifier.py' << 'SKETCHLAB_EOF'
"""
train_classifier.py
Stage 1 (mirrors paper Section III-A): fine-tune the pretrained face network
so it recognizes each subject as the same identity regardless of whether the
input is their photo or their sketch. Trained as ordinary classification,
one class per training identity.

Run: python3 train_classifier.py
Produces: stage1_model.pth

CLI options (used by run_ablation.py to sweep configs):
  --augment / --no-augment   toggle training-time augmentation (default: on)
  --epochs N                  number of epochs (default: 15)
  --output PATH               checkpoint filename (default: stage1_model.pth)
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_utils import get_device, load_image_tensor, build_backbone, EMBEDDING_DIM

BATCH_SIZE = 16
LEARNING_RATE = 1e-4


class IdentityDataset(Dataset):
    """Each identity contributes two samples: its photo and its sketch,
    both labeled with the same class index."""

    def __init__(self, pairs_json_path, augment):
        with open(pairs_json_path) as f:
            identities = json.load(f)
        self.samples = []
        for entry in identities:
            label = entry["identity_id"]
            self.samples.append((entry["photo"], label))
            self.samples.append((entry["sketch"], label))
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = load_image_tensor(path, augment=self.augment)
        return tensor, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", default="stage1_model.pth")
    parser.add_argument("--train_pairs", default="train_pairs.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Config: augment={args.augment} epochs={args.epochs} output={args.output}")

    train_dataset = IdentityDataset(args.train_pairs, augment=args.augment)
    num_classes = len({label for _, label in train_dataset.samples})
    print(f"Training identities (classes): {num_classes}")
    print(f"Training samples (photos + sketches): {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    backbone = build_backbone(device)
    classifier_head = nn.Linear(EMBEDDING_DIM, num_classes).to(device)

    params = [p for p in backbone.parameters() if p.requires_grad] + list(classifier_head.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("\nStarting Stage 1 training (classification)...\n")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        classifier_head.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = backbone(images)
            logits = classifier_head(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        print(f"Epoch {epoch:2d}/{args.epochs} | loss={total_loss / total:.4f} | "
              f"train_acc={correct / total:.3f}")

    elapsed = time.time() - start
    print(f"\nStage 1 finished in {elapsed:.1f}s.")

    torch.save({"backbone_state_dict": backbone.state_dict()}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'train_triplet.py' << 'SKETCHLAB_EOF'
"""
train_triplet.py
Stage 2 (mirrors paper Section III-A, "tuned for verification using triplet
embedding"): fine-tune the Stage 1 model with triplet loss, so that distance
between embeddings directly reflects same/different identity — this is what
retrieval (evaluate.py, predict.py) actually relies on.

Anchor = a sketch, Positive = that identity's photo, Negative = a random
different identity's photo. Same triplet-distance idea as the paper's
verification tuning (Fig. 1, yellow box), minus the paper's use of multiple
synthetic sketches per anchor.

Run: python3 train_triplet.py
Produces: stage2_model.pth

CLI options:
  --augment / --no-augment    toggle training-time augmentation (default: on)
  --epochs N                   number of epochs (default: 15)
  --stage1_model PATH           input Stage 1 checkpoint (default: stage1_model.pth)
  --output PATH                 output checkpoint filename (default: stage2_model.pth)
"""

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_utils import get_device, load_image_tensor, build_backbone

BATCH_SIZE = 16
LEARNING_RATE = 1e-5  # lower than Stage 1 — fine-tuning, not learning from scratch
MARGIN = 0.3


class TripletDataset(Dataset):
    def __init__(self, pairs_json_path, augment):
        with open(pairs_json_path) as f:
            self.identities = json.load(f)
        self.augment = augment

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, idx):
        anchor_entry = self.identities[idx]
        negative_entry = random.choice([e for e in self.identities if e["identity_id"] != anchor_entry["identity_id"]])

        anchor = load_image_tensor(anchor_entry["sketch"], augment=self.augment)
        positive = load_image_tensor(anchor_entry["photo"], augment=self.augment)
        negative = load_image_tensor(negative_entry["photo"], augment=self.augment)
        return anchor, positive, negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--stage1_model", default="stage1_model.pth")
    parser.add_argument("--output", default="stage2_model.pth")
    parser.add_argument("--train_pairs", default="train_pairs.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Config: augment={args.augment} epochs={args.epochs} "
          f"stage1_model={args.stage1_model} output={args.output}")

    train_dataset = TripletDataset(args.train_pairs, augment=args.augment)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training identities: {len(train_dataset)}")

    backbone = build_backbone(device)
    checkpoint = torch.load(args.stage1_model, map_location=device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])

    params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    triplet_loss_fn = nn.TripletMarginLoss(margin=MARGIN)

    print("\nStarting Stage 2 training (triplet embedding)...\n")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        total_loss, n_batches = 0.0, 0

        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            emb_a = F.normalize(backbone(anchor), dim=1)
            emb_p = F.normalize(backbone(positive), dim=1)
            emb_n = F.normalize(backbone(negative), dim=1)

            loss = triplet_loss_fn(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch:2d}/{args.epochs} | triplet_loss={total_loss / n_batches:.4f}")

    elapsed = time.time() - start
    print(f"\nStage 2 finished in {elapsed:.1f}s.")

    torch.save({"backbone_state_dict": backbone.state_dict()}, args.output)
    print(f"Saved {args.output} — this is your final embedding model.")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'evaluate.py' << 'SKETCHLAB_EOF'
"""
evaluate.py
Reproduces the paper's Table I evaluation protocol: for each probe sketch
in the test set, rank all test-set gallery photos by embedding distance and
check whether the correct match appears in the top N. Reports Rank-1,
Rank-5, Rank-10, and Rank-20 matching rates.

Run: python3 evaluate.py
"""

import json

import torch
import torch.nn.functional as F

from model_utils import get_device, load_image_tensor, build_backbone

RANKS_TO_REPORT = [1, 5, 10, 20]


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open("test_pairs.json") as f:
        test_identities = json.load(f)
    print(f"Test identities: {len(test_identities)}")

    backbone = build_backbone(device)
    checkpoint = torch.load("stage2_model.pth", map_location=device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval()

    # Build the gallery: one embedding per test identity's PHOTO
    gallery_ids, gallery_embeddings = [], []
    with torch.no_grad():
        for entry in test_identities:
            tensor = load_image_tensor(entry["photo"]).unsqueeze(0).to(device)
            emb = F.normalize(backbone(tensor), dim=1)
            gallery_ids.append(entry["identity_id"])
            gallery_embeddings.append(emb)
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)  # [N, 512]

    # For each probe SKETCH, rank all gallery photos by distance
    ranks_achieved = []
    with torch.no_grad():
        for entry in test_identities:
            tensor = load_image_tensor(entry["sketch"]).unsqueeze(0).to(device)
            probe_emb = F.normalize(backbone(tensor), dim=1)

            distances = torch.norm(gallery_embeddings - probe_emb, dim=1)
            sorted_indices = torch.argsort(distances)
            sorted_ids = [gallery_ids[i] for i in sorted_indices]

            correct_rank = sorted_ids.index(entry["identity_id"]) + 1  # 1-indexed
            ranks_achieved.append(correct_rank)

    print("\nRank-N matching rate (paper's Table I metric):")
    for n in RANKS_TO_REPORT:
        if n > len(test_identities):
            continue
        hit_rate = sum(1 for r in ranks_achieved if r <= n) / len(ranks_achieved)
        print(f"  Rank-{n:<3d}: {hit_rate * 100:.2f}%")

    mean_rank = sum(ranks_achieved) / len(ranks_achieved)
    print(f"\nMean rank: {mean_rank:.2f} (out of {len(test_identities)} gallery photos)")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'predict.py' << 'SKETCHLAB_EOF'
"""
predict.py
The actual use case the paper targets: given a sketch, retrieve the most
likely matching photos from a gallery, ranked by similarity.

Run: python3 predict.py --sketch /path/to/sketch.jpg --top_k 5
By default searches the test-set gallery (test_pairs.json); pass --gallery
to point at a different pairs JSON file.
"""

import argparse
import json

import torch
import torch.nn.functional as F

from model_utils import get_device, load_image_tensor, build_backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch", required=True, help="Path to a query sketch image")
    parser.add_argument("--gallery", default="test_pairs.json", help="Pairs JSON file to search")
    parser.add_argument("--model", default="stage2_model.pth", help="Trained embedding model checkpoint")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    device = get_device()

    with open(args.gallery) as f:
        gallery_entries = json.load(f)

    backbone = build_backbone(device)
    checkpoint = torch.load(args.model, map_location=device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval()

    with torch.no_grad():
        query_tensor = load_image_tensor(args.sketch).unsqueeze(0).to(device)
        query_emb = F.normalize(backbone(query_tensor), dim=1)

        gallery_embeddings, gallery_paths = [], []
        for entry in gallery_entries:
            tensor = load_image_tensor(entry["photo"]).unsqueeze(0).to(device)
            emb = F.normalize(backbone(tensor), dim=1)
            gallery_embeddings.append(emb)
            gallery_paths.append(entry["photo"])

        gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
        distances = torch.norm(gallery_embeddings - query_emb, dim=1)
        sorted_indices = torch.argsort(distances)[: args.top_k]

    print(f"Top {args.top_k} matches for {args.sketch}:\n")
    for rank, idx in enumerate(sorted_indices, start=1):
        print(f"  Rank {rank}: {gallery_paths[idx]}  (distance={distances[idx]:.4f})")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'generate_synthetic_sketches.py' << 'SKETCHLAB_EOF'
"""
generate_synthetic_sketches.py

Substitutes for the paper's 3D Morphable Model synthesis (Section III-B),
which we cannot reproduce without a licensed model. Instead, we generate
K augmented variants of each test-set sketch using the same 2D augmentation
already used during training (flip, rotation, brightness/contrast jitter).

This is NOT equivalent to the paper's approach — the 3DMM varies actual
facial attributes (eyes, nose, mouth shape, weight, age, height, gender) in
a way that can plausibly correct for sketch distortions. 2D augmentation
only perturbs the existing image; it cannot invent a more accurate face
shape. Treat downstream multi-sketch fusion results as a test of "does
having several augmented views help at all," not a reproduction of DEEPS-M.

Run: python3 generate_synthetic_sketches.py --k 8
Produces:
  synthetic_sketches/<identity_id>_<variant>.jpg
  multisketch_pairs.json  (test identities + paths to their variants)
"""

import argparse
import json
import os
import random

from PIL import Image, ImageEnhance

IMAGE_SIZE = 160


def make_variant(sketch_path, seed):
    """Generate one augmented variant of a sketch image, saved at a fixed
    size. Same augmentation family as model_utils.load_image_tensor's
    training-time augmentation, but applied here to produce actual image
    files rather than tensors, since we need them as reusable artifacts."""
    random.seed(seed)
    img = Image.open(sketch_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=(128, 128, 128))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8,
                         help="Number of synthetic variants per sketch (paper's DEEPS-M uses 9)")
    parser.add_argument("--test_pairs", default="test_pairs.json")
    parser.add_argument("--output_dir", default="synthetic_sketches")
    args = parser.parse_args()

    with open(args.test_pairs) as f:
        test_identities = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    multisketch_entries = []
    for entry in test_identities:
        identity_id = entry["identity_id"]
        variant_paths = []
        for k in range(args.k):
            variant_img = make_variant(entry["sketch"], seed=identity_id * 1000 + k)
            variant_path = os.path.join(args.output_dir, f"{identity_id}_{k}.jpg")
            variant_img.save(variant_path)
            variant_paths.append(variant_path)

        multisketch_entries.append({
            "identity_id": identity_id,
            "photo": entry["photo"],
            "sketch": entry["sketch"],
            "synthetic_sketches": variant_paths,
        })

    with open("multisketch_pairs.json", "w") as f:
        json.dump(multisketch_entries, f, indent=2)

    print(f"Generated {args.k} variants for {len(test_identities)} test identities "
          f"({args.k * len(test_identities)} images total) in {args.output_dir}/")
    print("Saved multisketch_pairs.json")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'evaluate_multisketch.py' << 'SKETCHLAB_EOF'
"""
evaluate_multisketch.py

Substitute for the paper's DEEPS-M evaluation (Section III-C): for each
probe identity, compare EVERY synthetic sketch variant (plus the original)
against each gallery photo, and fuse by taking the BEST (minimum-distance)
match — "the best match among nine sketches and the original," per the
paper's method 2. We skip the paper's method 1 (median of top-49 matches
across 200 synthetic sketches) since our augmentation produces far fewer,
lower-diversity variants than their 3DMM does; comparing to only the
best-match fusion keeps the comparison honest about what we're actually
testing.

Prints both the single-sketch baseline (identical to evaluate.py) and the
multi-sketch fused result side by side, so you can see whether fusion
actually helps with THIS augmentation method — it may not, since 2D
augmentation doesn't correct facial-attribute distortions the way a 3DMM
does. Report whatever you find; a null result here is a legitimate finding.

Run: python3 generate_synthetic_sketches.py --k 8   (if not already run)
     python3 evaluate_multisketch.py
"""

import argparse
import json

import torch
import torch.nn.functional as F

from model_utils import get_device, load_image_tensor, build_backbone

RANKS_TO_REPORT = [1, 5, 10, 20]


def compute_gallery_embeddings(backbone, device, entries):
    gallery_ids, gallery_embeddings = [], []
    with torch.no_grad():
        for entry in entries:
            tensor = load_image_tensor(entry["photo"]).unsqueeze(0).to(device)
            emb = F.normalize(backbone(tensor), dim=1)
            gallery_ids.append(entry["identity_id"])
            gallery_embeddings.append(emb)
    return gallery_ids, torch.cat(gallery_embeddings, dim=0)


def rank_of_correct_match(distances, gallery_ids, correct_id):
    sorted_ids = [gallery_ids[i] for i in torch.argsort(distances)]
    return sorted_ids.index(correct_id) + 1


def report_ranks(label, ranks_achieved, n_test):
    print(f"\n{label}:")
    for n in RANKS_TO_REPORT:
        if n > n_test:
            continue
        hit_rate = sum(1 for r in ranks_achieved if r <= n) / len(ranks_achieved)
        print(f"  Rank-{n:<3d}: {hit_rate * 100:.2f}%")
    print(f"  Mean rank: {sum(ranks_achieved) / len(ranks_achieved):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multisketch_pairs", default="multisketch_pairs.json")
    parser.add_argument("--model", default="stage2_model.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    with open(args.multisketch_pairs) as f:
        entries = json.load(f)
    print(f"Test identities: {len(entries)}")

    backbone = build_backbone(device)
    checkpoint = torch.load(args.model, map_location=device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval()

    gallery_ids, gallery_embeddings = compute_gallery_embeddings(backbone, device, entries)

    single_sketch_ranks = []
    multisketch_ranks = []

    with torch.no_grad():
        for entry in entries:
            # --- Baseline: original sketch only (same as evaluate.py) ---
            orig_tensor = load_image_tensor(entry["sketch"]).unsqueeze(0).to(device)
            orig_emb = F.normalize(backbone(orig_tensor), dim=1)
            orig_distances = torch.norm(gallery_embeddings - orig_emb, dim=1)
            single_sketch_ranks.append(rank_of_correct_match(orig_distances, gallery_ids, entry["identity_id"]))

            # --- Multi-sketch fusion: best match among original + all variants ---
            all_sketch_paths = [entry["sketch"]] + entry["synthetic_sketches"]
            all_embs = []
            for path in all_sketch_paths:
                tensor = load_image_tensor(path).unsqueeze(0).to(device)
                all_embs.append(F.normalize(backbone(tensor), dim=1))
            all_embs = torch.cat(all_embs, dim=0)  # [K+1, 512]

            # distance from EVERY sketch variant to EVERY gallery photo, then
            # take the minimum across variants for each gallery photo
            # (the "best match among sketches" fusion rule)
            distances_per_variant = torch.cdist(all_embs, gallery_embeddings)  # [K+1, N_gallery]
            fused_distances = distances_per_variant.min(dim=0).values  # [N_gallery]

            multisketch_ranks.append(rank_of_correct_match(fused_distances, gallery_ids, entry["identity_id"]))

    report_ranks("Single-sketch baseline (original sketch only)", single_sketch_ranks, len(entries))
    report_ranks("Multi-sketch fusion (best match among original + variants)", multisketch_ranks, len(entries))

    improved = sum(1 for a, b in zip(single_sketch_ranks, multisketch_ranks) if b < a)
    worsened = sum(1 for a, b in zip(single_sketch_ranks, multisketch_ranks) if b > a)
    print(f"\nPer-identity comparison: fusion improved rank for {improved}/{len(entries)}, "
          f"worsened for {worsened}/{len(entries)}, unchanged for {len(entries) - improved - worsened}.")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'handcrafted_features.py' << 'SKETCHLAB_EOF'
"""
handcrafted_features.py

Substitute for LGMS (Galea & Farrugia's own earlier method: log-Gabor
filtering + multiscale local binary patterns + Spearman rank-order
correlation). LGMS itself is a separate paper's worth of engineering; full
reimplementation is out of scope here. Instead we use a simpler, well-known
hand-crafted descriptor — Histogram of Oriented Gradients (HOG) — combined
with the SAME distance metric LGMS uses (Spearman rank-order correlation),
so the fusion experiment (fusion_eval.py) still mirrors the paper's actual
fusion methodology even though the underlying feature is different.

Be precise in any writeup: this is "HOG+Spearman as an LGMS-style
hand-crafted baseline," not a reproduction of LGMS itself. LGMS reportedly
scores far higher (82.92% Rank-1 on CUFS-style data) than a plain HOG
descriptor should be expected to.
"""

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from skimage.feature import hog

IMAGE_SIZE = 128


def extract_hog_feature(path):
    """Extract a HOG descriptor from an image, working from grayscale so
    photos and sketches are compared in the same feature space."""
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img)
    feature = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feature


def spearman_distance(feature_a, feature_b):
    """1 - Spearman rank correlation, so 0 = identical rank pattern
    (perfect match) and larger values = less similar — same convention as
    the L2 distances used elsewhere in this project, so the two can be
    fused directly after min-max normalization."""
    correlation, _p_value = spearmanr(feature_a, feature_b)
    if np.isnan(correlation):
        correlation = 0.0
    return 1.0 - correlation
SKETCHLAB_EOF

cat > 'fusion_eval.py' << 'SKETCHLAB_EOF'
"""
fusion_eval.py

Mirrors the paper's System Fusion approach (Section III-D): combine two
independent recognition methods via min-max normalization + sum-of-scores,
the same fusion rule the paper uses for LGMS+DEEPS. Here we fuse:
  - our fine-tuned deep embedding model (stage2_model.pth)
  - the HOG+Spearman hand-crafted baseline (handcrafted_features.py,
    substituting for LGMS — see that file's docstring for why)

Prints three Rank-N tables side by side: deep-only, handcrafted-only, and
fused — so you can see whether fusion actually helps, exactly the kind of
comparison the paper makes in Table I.

Run: python3 fusion_eval.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from model_utils import get_device, load_image_tensor, build_backbone
from handcrafted_features import extract_hog_feature, spearman_distance

RANKS_TO_REPORT = [1, 5, 10, 20]


def min_max_normalize(matrix):
    """Normalize a distance matrix to [0, 1] using its own min/max, the
    same normalization the paper uses before summing two methods' scores
    (Section III-D references min-max + sum-of-scores fusion)."""
    lo, hi = matrix.min(), matrix.max()
    if hi - lo < 1e-8:
        return np.zeros_like(matrix)
    return (matrix - lo) / (hi - lo)


def report_ranks(label, ranks_achieved, n_test):
    print(f"\n{label}:")
    for n in RANKS_TO_REPORT:
        if n > n_test:
            continue
        hit_rate = sum(1 for r in ranks_achieved if r <= n) / len(ranks_achieved)
        print(f"  Rank-{n:<3d}: {hit_rate * 100:.2f}%")
    print(f"  Mean rank: {sum(ranks_achieved) / len(ranks_achieved):.2f}")


def ranks_from_distance_matrix(distance_matrix, gallery_ids, probe_ids):
    """distance_matrix: [n_probes, n_gallery]. Returns list of 1-indexed
    ranks at which each probe's correct gallery match was found."""
    ranks = []
    for i, correct_id in enumerate(probe_ids):
        order = np.argsort(distance_matrix[i])
        sorted_ids = [gallery_ids[j] for j in order]
        ranks.append(sorted_ids.index(correct_id) + 1)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs", default="test_pairs.json")
    parser.add_argument("--model", default="stage2_model.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    with open(args.test_pairs) as f:
        entries = json.load(f)
    n_test = len(entries)
    print(f"Test identities: {n_test}")

    gallery_ids = [e["identity_id"] for e in entries]
    probe_ids = [e["identity_id"] for e in entries]

    # ---------------------------------------------------------------
    # 1. Deep embedding distances (stage2_model.pth), same as evaluate.py
    # ---------------------------------------------------------------
    backbone = build_backbone(device)
    checkpoint = torch.load(args.model, map_location=device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval()

    with torch.no_grad():
        gallery_embs = torch.cat([
            F.normalize(backbone(load_image_tensor(e["photo"]).unsqueeze(0).to(device)), dim=1)
            for e in entries
        ], dim=0)
        probe_embs = torch.cat([
            F.normalize(backbone(load_image_tensor(e["sketch"]).unsqueeze(0).to(device)), dim=1)
            for e in entries
        ], dim=0)
        deep_distance_matrix = torch.cdist(probe_embs, gallery_embs).cpu().numpy()

    # ---------------------------------------------------------------
    # 2. Hand-crafted (HOG+Spearman) distances
    # ---------------------------------------------------------------
    print("Extracting HOG features...")
    gallery_hog = [extract_hog_feature(e["photo"]) for e in entries]
    probe_hog = [extract_hog_feature(e["sketch"]) for e in entries]

    hand_distance_matrix = np.zeros((n_test, n_test))
    for i in range(n_test):
        for j in range(n_test):
            hand_distance_matrix[i, j] = spearman_distance(probe_hog[i], gallery_hog[j])

    # ---------------------------------------------------------------
    # 3. Fusion: min-max normalize each, then sum-of-scores
    # ---------------------------------------------------------------
    deep_norm = min_max_normalize(deep_distance_matrix)
    hand_norm = min_max_normalize(hand_distance_matrix)
    fused_matrix = deep_norm + hand_norm

    # ---------------------------------------------------------------
    # 4. Rank-N for all three
    # ---------------------------------------------------------------
    deep_ranks = ranks_from_distance_matrix(deep_distance_matrix, gallery_ids, probe_ids)
    hand_ranks = ranks_from_distance_matrix(hand_distance_matrix, gallery_ids, probe_ids)
    fused_ranks = ranks_from_distance_matrix(fused_matrix, gallery_ids, probe_ids)

    report_ranks("Deep embedding only (stage2_model.pth)", deep_ranks, n_test)
    report_ranks("Hand-crafted only (HOG+Spearman, LGMS substitute)", hand_ranks, n_test)
    report_ranks("Fused (deep + hand-crafted, min-max + sum-of-scores)", fused_ranks, n_test)


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'extend_gallery.py' << 'SKETCHLAB_EOF'
"""
extend_gallery.py

Substitute for the paper's extended-gallery simulation (Section III-D),
which pads the test gallery with photos of thousands of extra subjects
from MEDS-II, FRGC v2.0, Multi-PIE, and FEI — datasets that require
individual registration/licensing. We use LFW (Labeled Faces in the Wild)
instead: a long-standing, freely available academic face dataset, to add
"distractor" photos that dilute the gallery the same way — making retrieval
meaningfully harder and closer to a real mugshot-database scale.

These distractor photos have NO matching sketch (obviously — nobody drew a
forensic sketch of a randomly chosen LFW subject). They only ever appear as
gallery/distractor entries, never as probes, exactly mirroring how the
paper's extension subjects work.

Run: python3 extend_gallery.py --n_distractors 500
Produces:
  distractor_photos/<n>.jpg
  extended_gallery.json   (test set photos + distractor photos, with a
                            "has_sketch" flag distinguishing the two)
"""

import argparse
import json
import os

from PIL import Image


def fetch_via_sklearn(n_needed):
    """Primary path: scikit-learn's built-in LFW fetcher, which downloads
    from a stable, sklearn-maintained mirror. Requires internet access."""
    from sklearn.datasets import fetch_lfw_people

    print("Downloading LFW via scikit-learn (this can take a few minutes "
          "the first time)...")
    lfw = fetch_lfw_people(min_faces_per_person=1, resize=1.0, color=True)
    images = lfw.images  # [n_samples, h, w, 3], float in [0, 255]
    print(f"LFW dataset loaded: {len(images)} images available.")

    if len(images) < n_needed:
        print(f"WARNING: LFW only has {len(images)} images, fewer than the "
              f"{n_needed} requested — using all available.")
        n_needed = len(images)

    # Sample evenly across the dataset for identity diversity rather than
    # just taking the first N (which tend to cluster on a few celebrities
    # with many photos each in LFW).
    step = max(1, len(images) // n_needed)
    selected_indices = list(range(0, len(images), step))[:n_needed]
    return [images[i] for i in selected_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_distractors", type=int, default=500,
                         help="Number of LFW distractor photos to add to the gallery")
    parser.add_argument("--test_pairs", default="test_pairs.json")
    parser.add_argument("--output_dir", default="distractor_photos")
    parser.add_argument("--lfw_dir", default=None,
                         help="If sklearn's downloader fails (no internet), manually "
                              "download http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz, "
                              "extract it, and point this at the extracted folder instead.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.test_pairs) as f:
        test_identities = json.load(f)

    distractor_paths = []

    if args.lfw_dir:
        print(f"Using manually-downloaded LFW at {args.lfw_dir}")
        found = []
        for root, _dirs, files in os.walk(args.lfw_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    found.append(os.path.join(root, fname))
                    if len(found) >= args.n_distractors:
                        break
            if len(found) >= args.n_distractors:
                break
        for i, src_path in enumerate(found):
            dst_path = os.path.join(args.output_dir, f"{i}.jpg")
            Image.open(src_path).convert("RGB").save(dst_path)
            distractor_paths.append(dst_path)
    else:
        try:
            images = fetch_via_sklearn(args.n_distractors)
            for i, arr in enumerate(images):
                img = Image.fromarray(arr.astype("uint8"))
                dst_path = os.path.join(args.output_dir, f"{i}.jpg")
                img.save(dst_path)
                distractor_paths.append(dst_path)
        except Exception as e:
            print(f"\nERROR downloading LFW via scikit-learn: {e}")
            print("If this is a network issue, manually download "
                  "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz, extract it, "
                  "and rerun with --lfw_dir /path/to/extracted/lfw-deepfunneled")
            return

    # Build the extended gallery: test-set photos (have a matching sketch,
    # used as both probe target and gallery entry) + distractor photos
    # (gallery-only, no sketch).
    extended_gallery = []
    for entry in test_identities:
        extended_gallery.append({
            "identity_id": entry["identity_id"],
            "photo": entry["photo"],
            "has_sketch": True,
        })
    for i, path in enumerate(distractor_paths):
        extended_gallery.append({
            "identity_id": f"distractor_{i}",
            "photo": path,
            "has_sketch": False,
        })

    with open("extended_gallery.json", "w") as f:
        json.dump(extended_gallery, f, indent=2)

    print(f"\nExtended gallery built: {len(test_identities)} real test-set photos + "
          f"{len(distractor_paths)} LFW distractors = {len(extended_gallery)} total.")
    print(f"That's a {len(extended_gallery) / max(len(test_identities), 1):.1f}x larger "
          f"gallery than the plain test set — a harder, more realistic retrieval task.")
    print("Saved extended_gallery.json")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'evaluate_extended_gallery.py' << 'SKETCHLAB_EOF'
"""
evaluate_extended_gallery.py

Same Rank-N evaluation as evaluate.py, but retrieving against the extended
gallery from extend_gallery.py (test-set photos + LFW distractors) instead
of just the test-set photos. This is the harder, more realistic version of
the retrieval task — closer to the paper's simulated mugshot-database
evaluation, though smaller in scale (hundreds, not thousands, of
distractors, since that's what's practical to compute on Colab's free GPU
in a reasonable time).

Run: python3 extend_gallery.py --n_distractors 500   (if not already run)
     python3 evaluate_extended_gallery.py
"""

import argparse
import json

import torch
import torch.nn.functional as F

from model_utils import get_device, load_image_tensor, build_backbone

RANKS_TO_REPORT = [1, 5, 10, 20, 50]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended_gallery", default="extended_gallery.json")
    parser.add_argument("--model", default="stage2_model.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    with open(args.extended_gallery) as f:
        gallery_entries = json.load(f)

    probe_entries = [e for e in gallery_entries if e["has_sketch"]]
    # probes need their sketch path — re-derive from test_pairs.json since
    # extended_gallery.json only stores photos
    with open("test_pairs.json") as f:
        test_identities = json.load(f)
    sketch_by_id = {e["identity_id"]: e["sketch"] for e in test_identities}

    print(f"Gallery size: {len(gallery_entries)} "
          f"({len(probe_entries)} real + {len(gallery_entries) - len(probe_entries)} distractors)")
    print(f"Probes: {len(probe_entries)}")

    backbone = build_backbone(device)
    checkpoint = torch.load(args.model, map_location=device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval()

    gallery_ids, gallery_embeddings = [], []
    with torch.no_grad():
        for entry in gallery_entries:
            tensor = load_image_tensor(entry["photo"]).unsqueeze(0).to(device)
            emb = F.normalize(backbone(tensor), dim=1)
            gallery_ids.append(entry["identity_id"])
            gallery_embeddings.append(emb)
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)

    ranks_achieved = []
    with torch.no_grad():
        for entry in probe_entries:
            sketch_path = sketch_by_id[entry["identity_id"]]
            tensor = load_image_tensor(sketch_path).unsqueeze(0).to(device)
            probe_emb = F.normalize(backbone(tensor), dim=1)

            distances = torch.norm(gallery_embeddings - probe_emb, dim=1)
            sorted_indices = torch.argsort(distances)
            sorted_ids = [gallery_ids[i] for i in sorted_indices]

            correct_rank = sorted_ids.index(entry["identity_id"]) + 1
            ranks_achieved.append(correct_rank)

    print(f"\nRank-N matching rate against EXTENDED gallery "
          f"({len(gallery_entries)} photos):")
    for n in RANKS_TO_REPORT:
        if n > len(gallery_entries):
            continue
        hit_rate = sum(1 for r in ranks_achieved if r <= n) / len(ranks_achieved)
        print(f"  Rank-{n:<3d}: {hit_rate * 100:.2f}%")

    mean_rank = sum(ranks_achieved) / len(ranks_achieved)
    print(f"\nMean rank: {mean_rank:.2f} (out of {len(gallery_entries)} gallery photos)")
    print("\nCompare this to evaluate.py's numbers on the plain (smaller) gallery — "
          "expect Rank-N to drop here, since there are more distractor photos to be "
          "confused with. That drop is itself the point: it shows how much the "
          "paper's small-gallery numbers can overstate real-world performance.")


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

cat > 'pipeline.py' << 'SKETCHLAB_EOF'
"""
pipeline.py

The single unified entry point implementing the full diagram end-to-end,
instead of running evaluate.py / evaluate_multisketch.py / fusion_eval.py /
evaluate_extended_gallery.py separately and combining results by hand.

Mirrors the diagram exactly:
  Photo+sketch pair -> [augmentation, substituting 3DMM] -> fine-tune backbone
  -> trained embedding model (our DEEPS-equivalent)
      -> "viewed" sketches: use the model directly (one embedding, one distance)
      -> "forensic" sketches: generate K variants, fuse by best-match
  -> [both branches merge] -> fuse with hand-crafted method (LGMS substitute)
     via min-max normalization + sum-of-scores
  -> ranked match list

Three subcommands:

  python3 pipeline.py train
      Runs prepare_data.py -> train_classifier.py -> train_triplet.py in
      order. Equivalent to Steps 6-8 in the README, run as one command.

  python3 pipeline.py match --sketch PATH --mode viewed|forensic --top_k 5
      Runs ONE sketch through the full diagram and prints the final ranked
      match list. This is the "give me an answer for this sketch" entry
      point — replaces predict.py with the full fused pipeline instead of
      deep-embedding-only.

  python3 pipeline.py evaluate --mode viewed|forensic [--gallery test_pairs.json | extended_gallery.json]
      Runs the full pipeline over every test identity and reports Rank-N —
      the single number that reflects the WHOLE diagram working together,
      not each piece in isolation.

Run: python3 pipeline.py --help
"""

import argparse
import json
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from model_utils import get_device, load_image_tensor, build_backbone
from handcrafted_features import extract_hog_feature, spearman_distance
from generate_synthetic_sketches import make_variant

RANKS_TO_REPORT = [1, 5, 10, 20, 50]


# ---------------------------------------------------------------------------
# TRAIN subcommand — orchestrates the existing, already-tested scripts
# ---------------------------------------------------------------------------

def run_train():
    steps = [
        ["python3", "prepare_data.py"],
        ["python3", "train_classifier.py"],
        ["python3", "train_triplet.py"],
    ]
    for cmd in steps:
        print(f"\n{'=' * 60}\nRunning: {' '.join(cmd)}\n{'=' * 60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n'{' '.join(cmd)}' failed (exit code {result.returncode}). Stopping.")
            sys.exit(1)
    print("\nTraining pipeline complete: stage1_model.pth and stage2_model.pth are ready.")


# ---------------------------------------------------------------------------
# Shared pipeline internals — used by both `match` and `evaluate`
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, model_path="stage2_model.pth", k_variants=8):
        self.device = get_device()
        print(f"Using device: {self.device}")

        self.backbone = build_backbone(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        self.backbone.eval()

        self.k_variants = k_variants
        self._embedding_cache = {}
        self._hog_cache = {}

    def _embed(self, path):
        if path not in self._embedding_cache:
            with torch.no_grad():
                tensor = load_image_tensor(path).unsqueeze(0).to(self.device)
                self._embedding_cache[path] = F.normalize(self.backbone(tensor), dim=1)
        return self._embedding_cache[path]

    def _hog(self, path):
        if path not in self._hog_cache:
            self._hog_cache[path] = extract_hog_feature(path)
        return self._hog_cache[path]

    def deep_distance_to_gallery(self, sketch_path, mode, gallery_embeddings):
        """Diagram's branch: 'viewed' uses the model directly; 'forensic'
        generates K variants and fuses via best-match (min distance)."""
        if mode == "viewed":
            probe_emb = self._embed(sketch_path)
            return torch.cdist(probe_emb, gallery_embeddings).squeeze(0)

        elif mode == "forensic":
            with torch.no_grad():
                variant_embs = [self._embed(sketch_path)]
                for k in range(self.k_variants):
                    variant_img = make_variant(sketch_path, seed=hash(sketch_path) % 100000 + k)
                    tensor_arr = np.array(variant_img).astype(np.float32)
                    tensor = torch.from_numpy(tensor_arr).permute(2, 0, 1)
                    tensor = ((tensor - 127.5) / 128.0).unsqueeze(0).to(self.device)
                    variant_embs.append(F.normalize(self.backbone(tensor), dim=1))
                variant_embs = torch.cat(variant_embs, dim=0)  # [K+1, 512]
            distances_per_variant = torch.cdist(variant_embs, gallery_embeddings)  # [K+1, N]
            return distances_per_variant.min(dim=0).values

        else:
            raise ValueError(f"Unknown mode: {mode} (must be 'viewed' or 'forensic')")

    def handcrafted_distance_to_gallery(self, sketch_path, gallery_photo_paths):
        probe_feat = self._hog(sketch_path)
        return np.array([
            spearman_distance(probe_feat, self._hog(photo_path))
            for photo_path in gallery_photo_paths
        ])

    def match(self, sketch_path, gallery_ids, gallery_photo_paths, gallery_embeddings, mode):
        """Full diagram, single probe: deep branch (mode-dependent) + hand-crafted
        branch -> min-max normalize each -> sum -> ranked match list."""
        deep_dist = self.deep_distance_to_gallery(sketch_path, mode, gallery_embeddings).cpu().numpy()
        hand_dist = self.handcrafted_distance_to_gallery(sketch_path, gallery_photo_paths)

        deep_norm = _min_max_normalize(deep_dist)
        hand_norm = _min_max_normalize(hand_dist)
        fused = deep_norm + hand_norm

        order = np.argsort(fused)
        return [(gallery_ids[i], float(fused[i])) for i in order]


def _min_max_normalize(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _load_gallery(gallery_path, pipeline):
    """Supports both test_pairs.json (every entry has a sketch, used as
    both probe and gallery) and extended_gallery.json (has_sketch flag,
    distractors are gallery-only)."""
    with open(gallery_path) as f:
        entries = json.load(f)

    if entries and "has_sketch" in entries[0]:
        with open("test_pairs.json") as f:
            sketch_by_id = {e["identity_id"]: e["sketch"] for e in json.load(f)}
        gallery_ids = [e["identity_id"] for e in entries]
        gallery_photo_paths = [e["photo"] for e in entries]
        probe_entries = [
            {"identity_id": e["identity_id"], "sketch": sketch_by_id[e["identity_id"]]}
            for e in entries if e["has_sketch"]
        ]
    else:
        gallery_ids = [e["identity_id"] for e in entries]
        gallery_photo_paths = [e["photo"] for e in entries]
        probe_entries = [{"identity_id": e["identity_id"], "sketch": e["sketch"]} for e in entries]

    with torch.no_grad():
        gallery_embeddings = torch.cat([pipeline._embed(p) for p in gallery_photo_paths], dim=0)

    return gallery_ids, gallery_photo_paths, gallery_embeddings, probe_entries


# ---------------------------------------------------------------------------
# MATCH subcommand — one sketch in, ranked match list out
# ---------------------------------------------------------------------------

def run_match(args):
    pipeline = Pipeline(model_path=args.model, k_variants=args.k)
    gallery_ids, gallery_photo_paths, gallery_embeddings, _probes = _load_gallery(args.gallery, pipeline)

    results = pipeline.match(args.sketch, gallery_ids, gallery_photo_paths, gallery_embeddings, args.mode)

    print(f"\nMode: {args.mode} | Gallery size: {len(gallery_ids)}")
    print(f"Top {args.top_k} matches for {args.sketch}:\n")
    for rank, (identity_id, distance) in enumerate(results[:args.top_k], start=1):
        photo_path = gallery_photo_paths[gallery_ids.index(identity_id)]
        print(f"  Rank {rank}: identity={identity_id}  photo={photo_path}  fused_distance={distance:.4f}")


# ---------------------------------------------------------------------------
# EVALUATE subcommand — full pipeline over every test identity, Rank-N
# ---------------------------------------------------------------------------

def run_evaluate(args):
    pipeline = Pipeline(model_path=args.model, k_variants=args.k)
    gallery_ids, gallery_photo_paths, gallery_embeddings, probe_entries = _load_gallery(args.gallery, pipeline)

    print(f"Mode: {args.mode} | Gallery size: {len(gallery_ids)} | Probes: {len(probe_entries)}")

    ranks_achieved = []
    for i, probe in enumerate(probe_entries):
        results = pipeline.match(probe["sketch"], gallery_ids, gallery_photo_paths, gallery_embeddings, args.mode)
        ranked_ids = [identity_id for identity_id, _distance in results]
        ranks_achieved.append(ranked_ids.index(probe["identity_id"]) + 1)
        if (i + 1) % 10 == 0 or (i + 1) == len(probe_entries):
            print(f"  Processed {i + 1}/{len(probe_entries)} probes...")

    print(f"\nFull-pipeline Rank-N matching rate (mode={args.mode}, "
          f"gallery={args.gallery}):")
    for n in RANKS_TO_REPORT:
        if n > len(gallery_ids):
            continue
        hit_rate = sum(1 for r in ranks_achieved if r <= n) / len(ranks_achieved)
        print(f"  Rank-{n:<3d}: {hit_rate * 100:.2f}%")
    print(f"  Mean rank: {sum(ranks_achieved) / len(ranks_achieved):.2f}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run prepare_data -> train_classifier -> train_triplet in sequence")

    match_parser = subparsers.add_parser("match", help="Run one sketch through the full pipeline")
    match_parser.add_argument("--sketch", required=True)
    match_parser.add_argument("--mode", choices=["viewed", "forensic"], default="viewed")
    match_parser.add_argument("--gallery", default="test_pairs.json")
    match_parser.add_argument("--model", default="stage2_model.pth")
    match_parser.add_argument("--k", type=int, default=8, help="Variants to generate in forensic mode")
    match_parser.add_argument("--top_k", type=int, default=5)

    eval_parser = subparsers.add_parser("evaluate", help="Run the full pipeline over every test identity")
    eval_parser.add_argument("--mode", choices=["viewed", "forensic"], default="viewed")
    eval_parser.add_argument("--gallery", default="test_pairs.json")
    eval_parser.add_argument("--model", default="stage2_model.pth")
    eval_parser.add_argument("--k", type=int, default=8, help="Variants to generate in forensic mode")

    args = parser.parse_args()

    if args.command == "train":
        run_train()
    elif args.command == "match":
        run_match(args)
    elif args.command == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
SKETCHLAB_EOF

chmod +x download_data.sh
echo ""
echo "All 18 project files created:"
ls -la
echo ""
echo "Next: follow README.md starting at STEP 1."
