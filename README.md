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

source venv/bin/activate && python3 predict.py --sketch data/cufs/sketches/f-039-01-sz1.jpg --top_k 5
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
