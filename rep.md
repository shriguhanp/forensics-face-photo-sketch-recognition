# Forensic Face Photo-Sketch Recognition System
## Micro-Project Report — B.Tech (Computer Science & Engineering)

**Project Title:** Forensic Face Photo-Sketch Recognition Using Deep Transfer Learning and Score-Level Fusion

**Reference Paper:** Galea & Farrugia, *"Forensic Face Photo-Sketch Recognition Using a Deep Learning-Based Architecture"*, IEEE Transactions on Information Forensics and Security (TIFS), 2025.

---

## TABLE OF CONTENTS

1. Introduction
2. Literature Review
3. Problem Statement
4. Objectives
5. System Architecture
6. Methodology
7. Implementation
8. Results and Discussion
9. Conclusion
10. References
11. Appendix – I: Sample Inputs, Outputs and Implementation Details
12. Appendix – II: Additional Results and System Configuration

---

## 1. INTRODUCTION

Forensic sketch-to-photo face recognition is a critical component of modern law enforcement workflows. When photographic evidence is unavailable — such as in the absence of CCTV footage — trained forensic artists produce hand-drawn composite sketches based on witness accounts. Matching these sketches to an existing mugshot or criminal database is an extremely challenging task due to the severe modality gap that exists between the two image types. Photographs are rich in texture, colour, and photometric depth, while forensic sketches consist of sparse structural lines and edge information, often with geometric distortions introduced by the subjective nature of the drawing process.

This micro-project implements a complete, end-to-end forensic sketch recognition pipeline inspired by the core transfer-learning approach described in the IEEE TIFS 2025 reference paper. The system employs a two-stage fine-tuned deep learning backbone (InceptionResNetV1 pretrained on VGGFace2) to extract 512-dimensional facial embeddings, and complements these with Histogram of Oriented Gradients (HOG) structural features. The two similarity scores are fused at the score level using min-max normalization and summation, and the gallery photographs are ranked accordingly to produce a Top-K candidate list for forensic investigators.

---

## 2. LITERATURE REVIEW

The problem of photo-sketch face recognition has been addressed through multiple generations of approaches:

- **Hand-crafted feature methods** such as Locally Linear Embedding (LLE), Markov Random Fields (MRF), and Multiscale Local Binary Patterns (MLBP) attempted to bridge the modality gap through structural descriptors. While computationally lightweight, they suffered from low accuracy on real forensic sketches.

- **Synthesis-based approaches** using Generative Adversarial Networks (GANs) aimed to convert a sketch into a photo before recognition. However, this approach frequently introduces identity-altering hallucinations — generating texturally plausible but semantically incorrect faces — which is unacceptable in a forensic legal context.

- **Deep embedding approaches** map heterogeneous modalities into a shared latent space using large-scale pre-trained models. The FaceNet architecture and Triplet Loss metric learning demonstrated that a compact embedding space can be learned where same-identity pairs cluster together.

- **Galea & Farrugia (IEEE TIFS, 2025)** proposed a deep learning-based forensic face recognition system that combined a pretrained deep face recognition backbone with multi-sketch test-time fusion and a log-Gabor + MLBP + Spearman (LGMS) hand-crafted score-level fusion method, achieving state-of-the-art results on the CUFS dataset.

This project draws its core pipeline architecture from the above paper, while making practical substitutions: InceptionResNetV1 on VGGFace2 replaces the paper's VGG-Face backbone (which is not freely redistributable), and HOG+Spearman replaces the LGMS method (which is itself a separate paper's contribution). No claim is made that 3DMM synthesis or LGMS fusion from the paper is implemented as originally described.

---

## 3. PROBLEM STATEMENT

Standard face recognition systems are designed and optimized for photographic inputs. They perform poorly — or fail entirely — when presented with forensic sketches as probe images. The modality gap between photographs and hand-drawn sketches manifests as:

1. A **textural mismatch**: Photos contain colour, shadow, and surface texture; sketches contain only oriented edge strokes.
2. **Geometric distortion**: Forensic sketches drawn from witness memory introduce proportional errors in facial geometry (e.g., incorrect eye spacing, exaggerated jawlines).
3. **Intensity non-linearity**: The pixel intensity relationship between a photo and its corresponding sketch is non-linear and identity-dependent.

Existing solutions either fail to handle the modality gap or introduce unacceptable identity errors. The goal of this project is to implement a practical, reproducible forensic sketch recognition pipeline that bridges this modality gap using deep transfer learning and multi-feature score-level fusion, returning a ranked shortlist of candidate identities to assist forensic investigators.

---

## 4. OBJECTIVES

- To implement a cross-modal face recognition pipeline that accepts a forensic sketch as input and retrieves ranked matching photographs from a gallery database.
- To apply two-stage transfer learning on the CUFS dataset using an InceptionResNetV1 backbone pretrained on VGGFace2.
- To integrate HOG-based structural feature extraction as a complementary hand-crafted descriptor.
- To perform score-level fusion of deep embedding similarity and HOG structural similarity.
- To evaluate the system using the Rank-N matching rate metric as defined in the reference IEEE TIFS 2025 paper.
- To demonstrate the effectiveness of multi-feature fusion through ablation studies.

---

## 5. SYSTEM ARCHITECTURE

The system follows a sequential pipeline with a dual-branch feature extraction stage:

```
Forensic Sketch (Query)
        |
        v
+-------------------------------+
|   Preprocessing Engine        |
|   Resize -> 160x160 RGB       |
|   Normalize to [-1, 1]        |
+-------------+-----------------+
              |
       +------+------+
       v             v
+----------+   +------------------+
| Deep     |   | HOG Feature      |
| Branch   |   | Extraction       |
|          |   | (128x128,        |
| Inception|   | grayscale,       |
| ResNetV1 |   | 9 orientations,  |
| VGGFace2 |   | 8x8 cells,       |
| pretrained|  | 2x2 blocks,      |
| fine-tuned|  | L2-Hys norm)     |
|          |   +--------+---------+
| 512-D    |            |
| Embedding|            | Spearman
| (L2-norm)|            | Distance
+----+-----+            |
     | Euclidean        |
     | (L2) Distance    |
     +------+-----------+
            v
+-------------------------------+
|   Score-Level Fusion Engine   |
|   Min-Max Normalize each      |
|   Fused = Deep_norm + HOG_norm|
+-------------+-----------------+
              v
+-------------------------------+
|   Gallery Ranking             |
|   Sort by ascending fused     |
|   distance score              |
+-------------+-----------------+
              v
         Top-K Results
   (Ranked candidate identities)
```

---

## 6. METHODOLOGY

### 6.1 Dataset

The **CUHK Face Sketch (CUFS) dataset** was used for training and evaluation. It contains 606 subject pairs, each comprising one frontal face photograph and one artist-drawn sketch. The dataset was obtained from Kaggle and split into training and test sets using `prepare_data.py` and `pairing.py`.

### 6.2 Preprocessing

All images — both photographs and sketches — are preprocessed identically using the following pipeline (implemented in `model_utils.py`):

1. **Colour space conversion**: Image is opened and converted to RGB (3-channel) regardless of source format.
2. **Resizing**: Image is resized to 160×160 pixels, matching the expected input size of InceptionResNetV1.
3. **Normalization**: Pixel values are standardized using the formula: `(pixel - 127.5) / 128.0`, mapping values to approximately the `[-1, 1]` range.
4. **Tensor conversion**: The NumPy HWC array is converted to a PyTorch CHW tensor.

Training images additionally undergo optional random augmentation (horizontal flip, small rotations up to ±10°, brightness/contrast jitter) to serve as a practical 2D substitute for the 3DMM-based synthetic augmentation described in the reference paper.

### 6.3 Deep Feature Extraction — InceptionResNetV1

The core deep learning backbone is **InceptionResNetV1** loaded with weights pretrained on the **VGGFace2** dataset (approximately 3.3 million face images, over 9,000 identities). The model is used without its classification head (`classify=False`), outputting a 512-dimensional continuous embedding vector.

**Stage 1 — Classification Fine-Tuning (`train_classifier.py`):**
- The photo and sketch of each identity are assigned the same class label.
- A fully connected classification layer is added on top of the backbone.
- Loss function: Cross-Entropy Loss.
- Optimizer: Adam, learning rate = 1×10⁻⁴.
- Epochs: 15.
- Output: `stage1_model.pth`.

**Stage 2 — Triplet Margin Fine-Tuning (`train_triplet.py`):**
- Triplet samples: `(Anchor=Sketch, Positive=matching Photo, Negative=non-matching Photo)`.
- Loss function: Triplet Margin Loss, margin = 0.3.
- Optimizer: Adam, learning rate = 1×10⁻⁵ (reduced to avoid destroying Stage 1 progress).
- Epochs: 15.
- Output: `stage2_model.pth`.

At inference time, the final 512-D embedding vector is L2-normalized, and similarity is measured via Euclidean (L2) distance.

**Layer Freezing Strategy:**
The early layers of InceptionResNetV1 (`conv2d_1a` through `repeat_1`) are frozen during fine-tuning. These layers detect low-level edges and local texture patterns — features shared between photographs and sketches. Retraining them on the small CUFS dataset would cause catastrophic overfitting and destroy the rich VGGFace2 feature knowledge. Only `repeat_2`, `mixed_7a`, `repeat_3`, `block8`, `last_linear`, and `last_bn` are set as trainable.

### 6.4 HOG Feature Extraction

As a complementary structural descriptor, Histogram of Oriented Gradients (HOG) features are extracted using `handcrafted_features.py`. HOG focuses on the distribution of local gradient orientations — making it naturally suited for sketch images.

**Parameters used:**
- Image size: 128×128 (grayscale)
- Orientations: 9 bins
- Pixels per cell: 8×8
- Cells per block: 2×2
- Block normalization: L2-Hys

**Distance metric:** Spearman rank-order correlation distance (`1 - ρ`). Spearman correlation measures the monotonic rank relationship between two feature vectors, making it robust to the non-linear intensity differences between photographs and sketches.

### 6.5 Score-Level Fusion

The deep L2 distance and the HOG Spearman distance are on different numerical scales. Before combination, each distance vector (one value per gallery photo) is independently normalized to `[0, 1]` using min-max normalization:

```
normalized_score = (score - min) / (max - min)
fused_distance   = deep_norm + hog_norm
```

Gallery photographs are then sorted in ascending order of fused distance, and the Top-K candidates are returned as the recognition result.

### 6.6 Evaluation Metric

The system is evaluated using the **Rank-N matching rate**: the proportion of probe sketches for which the correct matching photo appears within the top-N positions of the ranked gallery. Rank-1, Rank-5, Rank-10, Rank-20, and Rank-50 rates are reported.

---

## 7. IMPLEMENTATION

The project is implemented entirely in Python 3 and organized into modular scripts. The full end-to-end pipeline is orchestrated by `pipeline.py`.

### 7.1 Project File Structure

```
Forensics/
├── pipeline.py                    # Unified entry point (train / match / evaluate)
├── model_utils.py                 # Preprocessing and backbone loading
├── train_classifier.py            # Stage 1: Cross-entropy classification fine-tuning
├── train_triplet.py               # Stage 2: Triplet margin fine-tuning
├── evaluate.py                    # Standalone Rank-N evaluation (deep only)
├── predict.py                     # Single-sketch query (deep only)
├── handcrafted_features.py        # HOG extraction + Spearman distance
├── fusion_eval.py                 # Ablation: deep-only / HOG-only / fused
├── pairing.py                     # Dataset photo-sketch pairing logic
├── prepare_data.py                # Train/test split -> JSON files
├── explore_data.py                # Dataset inspection and sample visualization
├── generate_synthetic_sketches.py # 2D augmentation-based sketch variants
├── extend_gallery.py              # LFW distractor gallery integration
├── evaluate_extended_gallery.py   # Evaluation against extended gallery
├── evaluate_multisketch.py        # Multi-sketch fusion evaluation
├── render_3dmm.py                 # Offline 3DMM software renderer (BFM 2019)
├── import_3dmm_synthetics.py      # 3DMM render ingestion script
├── requirements.txt               # Python dependencies
├── data/
|   └── cufs/
|       ├── photos/                # 190 face photographs (CUFS)
|       ├── sketches/              # 190 corresponding artist sketches
|       ├── photo/                 # Additional photo variants
|       └── sketch/                # Additional sketch variants
├── stage1_model.pth               # Saved Stage 1 fine-tuned model (~107 MB)
├── stage2_model.pth               # Saved Stage 2 fine-tuned model (~107 MB)
├── train_pairs.json               # Training identity-pair mappings
├── test_pairs.json                # Test set ground truth
├── extended_gallery.json          # Gallery with LFW distractors
└── multisketch_pairs.json         # Augmented sketch variant paths
```

---

## 8. RESULTS AND DISCUSSION

The system was trained and evaluated on the CUFS (CUHK Face Sketch) dataset. Two evaluation configurations were tested:

1. **Standard gallery** (`test_pairs.json`): Deep-only, HOG-only, and fused variants evaluated via `fusion_eval.py` and `pipeline.py evaluate`.
2. **Extended gallery** (`extended_gallery.json`): Gallery padded with LFW distractor photographs to simulate a more realistic, larger police database.

The extended gallery evaluation demonstrates that Rank-N metrics obtained on small closed-set galleries tend to overstate real-world performance. When the gallery is diluted with hundreds of distractor faces from the LFW dataset, the Rank-N rates decrease — an important and reportable finding about the scalability of the approach.

The fusion ablation (`fusion_eval.py`) allows direct comparison of deep-only, HOG-only, and combined (fused) similarity, validating whether multi-feature fusion provides a measurable benefit.

> **Note:** Specific numerical Rank-1/Rank-5/Rank-N accuracy values are not reported here, as they are subject to variation based on training run randomness, hardware, and exact data splits. The system should be executed using `python3 pipeline.py evaluate` and `python3 fusion_eval.py` to obtain actual evaluation numbers for a given trained model checkpoint.

---

## 9. CONCLUSION

*(See dedicated Conclusion section below, after Appendix II.)*

---

## 10. REFERENCES

*(See dedicated References section below.)*

---

---

# APPENDIX – I
## APPENDIX – I: SAMPLE INPUTS, OUTPUTS AND IMPLEMENTATION DETAILS

---

### A1.1 Sample Forensic Sketch Inputs and Gallery Photos

The CUFS (CUHK Face Sketch) dataset provides the primary input data for this system. Each subject is represented by:

- **One frontal face photograph** (stored under `data/cufs/photos/`): colour images of subjects taken under controlled lighting conditions.
- **One artist-drawn sketch** (stored under `data/cufs/sketches/`): black-and-white pencil sketches produced by a trained forensic artist while viewing the photograph.

**Example filename pairs from the dataset:**

| Subject ID | Photo Filename    | Sketch Filename        |
|------------|-------------------|------------------------|
| f-005      | f-005-01.jpg      | F2-005-01-sz1.jpg      |
| f-006      | f-006-01.jpg      | F2-006-01-sz1.jpg      |
| f-007      | f-007-01.jpg      | F2-007-01-sz1.jpg      |
| f-008      | f-008-01.jpg      | F2-008-01-sz1.jpg      |
| f-009      | f-009-01.jpg      | F2-009-01-sz1.jpg      |

The dataset contains 190 photo-sketch pairs in the primary working directories (`data/cufs/photos/` and `data/cufs/sketches/`). At the full CUFS dataset level, 606 total subject pairs are documented across all sub-directories.

**Nature of sketch inputs:** These are "viewed sketches" — drawn by an artist while looking at the photograph. In real forensic scenarios, sketches are drawn from witness memory and contain additional geometric distortion. The system provides a `--mode forensic` option in `pipeline.py` that uses test-time augmentation to simulate this added variability.

---

### A1.2 Preprocessing Example

The following illustrates the preprocessing applied to every image before it is fed into the deep learning model (implemented in `model_utils.py`, function `load_image_tensor`):

**Input:** Raw JPEG or PNG image (colour photograph or grayscale sketch)

**Step 1 — Colour space unification:**
```python
img = Image.open(path).convert("RGB")
```
Both photographs (colour) and sketches (grayscale) are converted to 3-channel RGB, ensuring consistent input dimensionality.

**Step 2 — Spatial resize:**
```python
img = img.resize((160, 160))
```
Images are resized to 160×160 pixels — the native input resolution expected by InceptionResNetV1.

**Step 3 — Normalization to [-1, 1]:**
```python
arr = np.array(img).astype(np.float32)
tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
tensor = (tensor - 127.5) / 128.0
```
This formula maps pixel values from `[0, 255]` to approximately `[-0.996, 1.0]`, consistent with VGGFace2-pretrained InceptionResNetV1 preprocessing.

**Output:** A `torch.Tensor` of shape `[3, 160, 160]` with dtype `float32`.

**Training-time augmentation** (applied only during training, never during evaluation):
```python
if augment:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)  # random horizontal flip
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, fillcolor=(128, 128, 128))  # random rotation +-10 deg
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
```

---

### A1.3 Feature Extraction Output — 512-D InceptionResNetV1 Embedding

After preprocessing, the image tensor is passed through the fine-tuned InceptionResNetV1 model (`stage2_model.pth`). The model outputs a 512-dimensional continuous latent vector. This vector is then L2-normalized before distance calculation.

**Key implementation code (from `predict.py`):**

```python
# Load backbone and checkpoint
backbone = build_backbone(device)
checkpoint = torch.load("stage2_model.pth", map_location=device)
backbone.load_state_dict(checkpoint["backbone_state_dict"])
backbone.eval()

# Embed a query sketch
with torch.no_grad():
    query_tensor = load_image_tensor(sketch_path).unsqueeze(0).to(device)
    query_emb = F.normalize(backbone(query_tensor), dim=1)
    # query_emb.shape -> torch.Size([1, 512])
```

**Description of the 512-D embedding:**
- Each of the 512 dimensions encodes a learned aspect of facial identity, including structural geometry, facial proportions, and cross-modal features learned during fine-tuning.
- L2 normalization ensures all embeddings lie on the surface of a unit hypersphere, making Euclidean distance and cosine similarity equivalent metrics.
- Distance between a sketch embedding and a photo embedding: smaller values indicate higher identity similarity.

**Backbone architecture and layer freezing (`model_utils.py`):**

```python
def build_backbone(device, freeze_until_block=6):
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)

    trainable_from = ["repeat_2", "mixed_7a", "repeat_3", "block8",
                      "last_linear", "last_bn"]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix)
                                  for prefix in trainable_from)
    return model
```

Layers `conv2d_1a` through `repeat_1` are frozen. They capture low-level edge and texture primitives equally present in both photographs and sketches; retraining them on the small CUFS dataset would lead to destructive overfitting.

---

### A1.4 HOG Feature Extraction — Structural Similarity Score

HOG features are extracted using `handcrafted_features.py`. The HOG descriptor encodes the distribution of local gradient orientations, making it ideal for sketch images where information is encoded in edge direction rather than pixel intensity.

**Complete HOG extraction and distance code:**

```python
from skimage.feature import hog
from scipy.stats import spearmanr
from PIL import Image
import numpy as np

IMAGE_SIZE = 128

def extract_hog_feature(path):
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img)
    feature = hog(
        arr,
        orientations=9,          # 9 orientation bins (0-180 degrees)
        pixels_per_cell=(8, 8),  # each cell is 8x8 pixels
        cells_per_block=(2, 2),  # normalization block: 2x2 cells
        block_norm="L2-Hys",     # standard Dalal/Triggs block normalization
        feature_vector=True,
    )
    return feature

def spearman_distance(feature_a, feature_b):
    # Returns 1 - rho; 0 = perfect match (same convention as L2 distance)
    correlation, _ = spearmanr(feature_a, feature_b)
    if np.isnan(correlation):
        correlation = 0.0
    return 1.0 - correlation
```

**HOG feature vector dimensionality** (for a 128×128 grayscale image with the above parameters):

| Quantity               | Calculation                          | Value       |
|------------------------|--------------------------------------|-------------|
| Number of cells        | (128/8) × (128/8) = 16 × 16         | 256 cells   |
| Number of blocks       | (16-1) × (16-1)                      | 225 blocks  |
| Descriptor per block   | 2 × 2 cells × 9 orientations         | 36 values   |
| Total HOG vector length| 225 × 36                             | **8,100**   |

**Why Spearman instead of Euclidean distance for HOG:**
Spearman rank-order correlation measures the monotonic relationship between two vectors — it cares about the *relative ordering* of feature values, not their absolute magnitudes. Since the intensity relationship between a photo and its corresponding sketch is highly non-linear (a dark shadow in a photo may be blank in the sketch), Spearman correlation is far more robust to this cross-modal intensity mismatch than L2 distance.

---

### A1.5 Sample Score-Fusion Output

The fusion pipeline is implemented in `pipeline.py`, method `Pipeline.match()`. The following illustrates the data flow for a single probe sketch against a gallery of N photographs:

**Step 1 — Obtain deep distances:**
```python
deep_dist = self.deep_distance_to_gallery(sketch_path, mode, gallery_embeddings).cpu().numpy()
# deep_dist: numpy array of shape [N], L2 distance to each gallery photo
```

**Step 2 — Obtain HOG Spearman distances:**
```python
hand_dist = self.handcrafted_distance_to_gallery(sketch_path, gallery_photo_paths)
# hand_dist: numpy array of shape [N], (1 - Spearman rho) for each gallery photo
```

**Step 3 — Min-max normalization:**
```python
def _min_max_normalize(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)

deep_norm = _min_max_normalize(deep_dist)  # Values in [0, 1]
hand_norm = _min_max_normalize(hand_dist)  # Values in [0, 1]
```

**Step 4 — Score fusion and ranking:**
```python
fused = deep_norm + hand_norm  # element-wise sum; range approx. [0, 2]
order = np.argsort(fused)      # ascending order -> best match first
```

**Sample console output (illustrative of output format):**
```
Mode: viewed | Gallery size: 38
Top 5 matches for data/cufs/sketches/F2-039-01-sz1.jpg:

  Rank 1: identity=f-039  photo=data/cufs/photos/f-039-01.jpg  fused_distance=0.0621
  Rank 2: identity=f-071  photo=data/cufs/photos/f-071-01.jpg  fused_distance=0.4738
  Rank 3: identity=f-055  photo=data/cufs/photos/f-055-01.jpg  fused_distance=0.5102
  Rank 4: identity=f-018  photo=data/cufs/photos/f-018-01.jpg  fused_distance=0.5349
  Rank 5: identity=f-022  photo=data/cufs/photos/f-022-01.jpg  fused_distance=0.5891
```

*(The identity labels and distances above illustrate the output format. Actual values will vary with the trained model checkpoint and dataset configuration.)*

---

### A1.6 Sample Top-K Recognition Results

The system is queried using either:

**Option A — Deep-only (`predict.py`):**
```bash
python3 predict.py --sketch data/cufs/sketches/F2-039-01-sz1.jpg --top_k 5
```

**Option B — Full fused pipeline (`pipeline.py`):**
```bash
python3 pipeline.py match \
    --sketch data/cufs/sketches/F2-039-01-sz1.jpg \
    --mode viewed \
    --top_k 5
```

**Full pipeline evaluation over entire test set:**
```bash
python3 pipeline.py evaluate --mode viewed
python3 pipeline.py evaluate --mode forensic
```

**Output format for full pipeline evaluation:**
```
Mode: viewed | Gallery size: 38 | Probes: 38
  Processed 10/38 probes...
  Processed 20/38 probes...
  Processed 30/38 probes...
  Processed 38/38 probes...

Full-pipeline Rank-N matching rate (mode=viewed, gallery=test_pairs.json):
  Rank-1  : [computed value]%
  Rank-5  : [computed value]%
  Rank-10 : [computed value]%
  Rank-20 : [computed value]%
  Mean rank: [computed value]
```

---

### A1.7 Key Implementation Snippets

**Identity pair data format (JSON):**
```json
[
  {
    "identity_id": "f-039",
    "photo": "data/cufs/photos/f-039-01.jpg",
    "sketch": "data/cufs/sketches/F2-039-01-sz1.jpg"
  }
]
```

**Triplet margin loss training core loop (`train_triplet.py`):**
```python
import torch.nn as nn

criterion = nn.TripletMarginLoss(margin=0.3)
optimizer = torch.optim.Adam(
    [p for p in backbone.parameters() if p.requires_grad],
    lr=1e-5
)
# Each batch: anchor=sketch embedding, positive=matching photo, negative=random other photo
loss = criterion(anchor_emb, positive_emb, negative_emb)
loss.backward()
optimizer.step()
```

**Vectorized gallery distance calculation:**
```python
# Stack all gallery embeddings into matrix [N, 512]
gallery_embeddings = torch.cat(gallery_embedding_list, dim=0)

# L2 distance: probe [1, 512] vs. all gallery [N, 512]
distances = torch.norm(gallery_embeddings - query_emb, dim=1)

# Equivalent vectorized form (more efficient for large N):
distances = torch.cdist(probe_emb, gallery_embeddings).squeeze(0)

sorted_indices = torch.argsort(distances)[:top_k]
```

---

### A1.8 Dataset Folder Structure

```
data/
└── cufs/
    ├── photos/           # 190 JPEG face photographs (primary working set)
    |   ├── f-005-01.jpg
    |   ├── f-006-01.jpg
    |   └── ... (naming pattern: f-{ID}-01.jpg)
    ├── sketches/         # 190 corresponding artist sketches
    |   ├── F2-005-01-sz1.jpg
    |   ├── F2-006-01-sz1.jpg
    |   └── ... (naming pattern: F2-{ID}-01-sz1.jpg)
    ├── photo/            # Additional/alternate photo variants
    ├── sketch/           # Additional/alternate sketch variants
    ├── original_sketch/  # Original unprocessed sketch images
    ├── cropped_sketch/   # Pre-cropped sketch variants
    ├── photo_points/     # Facial landmark files for photos
    └── sketch_points/    # Facial landmark files for sketches
```

---

---

# APPENDIX – II
## APPENDIX – II: ADDITIONAL RESULTS AND SYSTEM CONFIGURATION

---

### A2.1 Software Requirements

| Software / Library  | Version / Notes                          | Purpose                                            |
|---------------------|------------------------------------------|----------------------------------------------------|
| Python              | 3.12 (project venv) / 3.14 (system)     | Core programming language                          |
| PyTorch             | Latest stable (`pip install torch`)      | Deep learning framework, training, inference       |
| torchvision         | Latest stable                            | Image transforms used alongside PyTorch            |
| facenet-pytorch     | 2.6.0                                    | InceptionResNetV1 with VGGFace2 pretrained weights |
| scikit-image        | Latest stable                            | HOG feature extraction (`skimage.feature.hog`)     |
| SciPy               | Latest stable                            | Spearman rank correlation (`scipy.stats.spearmanr`)|
| scikit-learn        | Latest stable                            | LFW dataset fetching for extended gallery          |
| NumPy               | Latest stable                            | Array operations, normalization                    |
| Pillow (PIL)        | Latest stable                            | Image loading, resizing, augmentation              |
| Matplotlib (Agg)    | Latest stable                            | Headless 3DMM software rendering (offline only)    |
| h5py                | Latest stable                            | Reading Basel Face Model 2019 HDF5 file            |
| kaggle              | Latest stable                            | Programmatic CUFS dataset download from Kaggle     |

**Complete `requirements.txt`:**
```
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
h5py
```

**Environment Setup Commands:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision facenet-pytorch matplotlib pillow
pip install numpy scikit-learn scikit-image scipy kaggle h5py
```

---

### A2.2 Hardware Requirements

| Component           | Minimum                               | Used in This Project                         |
|---------------------|---------------------------------------|----------------------------------------------|
| Processor           | Any modern CPU (x86-64 or ARM)        | Apple Silicon M-series (MPS acceleration)    |
| RAM                 | 8 GB                                  | 16 GB recommended                            |
| Storage             | 5 GB free (models + dataset)          | ~600 MB dataset + 2x107 MB model checkpoints |
| GPU / Accelerator   | Optional (CPU fallback supported)     | Apple MPS (`torch.backends.mps`)             |
| Operating System    | Linux / macOS / Windows (with WSL)    | macOS (Apple Silicon)                        |
| Internet Connection | Required for initial dataset download | Used for Kaggle CUFS dataset download only   |

**Device detection code (`model_utils.py`):**
```python
def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

On Apple Silicon Macs, the MPS (Metal Performance Shaders) backend provides GPU-accelerated training and inference. On other platforms, the system automatically falls back to CPU.

---

### A2.3 Python Environment Details

The project uses a dedicated Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

The project venv is based on **Python 3.12**, with `facenet-pytorch 2.6.0` as a key pinned dependency. The system Python is 3.14.6.

**Verify installation:**
```bash
python3 -c "import torch, torchvision, facenet_pytorch, matplotlib, PIL, numpy, sklearn, kaggle; print('ALL OK')"
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

### A2.4 Model Configuration Summary

| Parameter                | Value                                                           |
|--------------------------|-----------------------------------------------------------------|
| Backbone Architecture    | InceptionResNetV1                                               |
| Pre-training Dataset     | VGGFace2 (~3.3M images, ~9,000 identities)                     |
| Pre-training via         | `facenet-pytorch` library (v2.6.0)                              |
| Fine-tuning Dataset      | CUFS (606 subjects, 1 photo + 1 sketch each)                   |
| Input Resolution         | 160×160 RGB                                                     |
| Input Normalization      | `(pixel - 127.5) / 128.0`                                      |
| Embedding Dimensionality | 512 (L2-normalized)                                             |
| Frozen Layers            | `conv2d_1a` through `repeat_1`                                 |
| Trainable Layers         | `repeat_2`, `mixed_7a`, `repeat_3`, `block8`, `last_linear`, `last_bn` |
| Stage 1 Loss             | Cross-Entropy Loss                                              |
| Stage 1 Optimizer        | Adam, LR = 1×10⁻⁴                                              |
| Stage 1 Epochs           | 15                                                              |
| Stage 1 Output           | `stage1_model.pth` (~107 MB)                                   |
| Stage 2 Loss             | Triplet Margin Loss, margin = 0.3                               |
| Stage 2 Optimizer        | Adam, LR = 1×10⁻⁵                                              |
| Stage 2 Epochs           | 15                                                              |
| Stage 2 Output           | `stage2_model.pth` (~107 MB)                                   |
| Triplet Sampling         | Anchor=Sketch, Positive=Match, Negative=Random Other            |
| Similarity Metric (Deep) | Euclidean (L2) distance                                         |

---

### A2.5 HOG Feature Configuration Summary

| Parameter             | Value                              |
|-----------------------|------------------------------------|
| Input size (HOG)      | 128×128 pixels                     |
| Colour space          | Grayscale (L channel)              |
| Orientation bins      | 9 (covering 0°–180°)              |
| Pixels per cell       | 8×8                                |
| Cells per block       | 2×2                                |
| Block normalization   | L2-Hys                             |
| Feature vector length | 8,100 dimensions                   |
| Distance metric       | Spearman rank correlation (1 - rho)|
| Library               | `skimage.feature.hog`              |

---

### A2.6 Dataset Configuration

| Parameter                  | Value                                          |
|----------------------------|------------------------------------------------|
| Primary Dataset            | CUFS (CUHK Face Sketch Database)               |
| Dataset Source             | Kaggle (via `kaggle` API)                      |
| Total Subject Pairs        | 606 (full CUFS)                                |
| Working photo-sketch pairs | 190 (primary `photos/` and `sketches/` dirs)   |
| Sketch Type                | "Viewed" (artist works from photo)             |
| Train/Test Split           | Via `prepare_data.py` (approx. 80/20)          |
| Extended Gallery           | LFW (Labeled Faces in the Wild) distractor photos |
| Extended Gallery Tool      | `extend_gallery.py --n_distractors 500`        |
| Gallery Format             | JSON files (`train_pairs.json`, `test_pairs.json`, `extended_gallery.json`) |

---

### A2.7 Fusion Configuration

| Parameter                    | Value                                                  |
|------------------------------|--------------------------------------------------------|
| Deep branch distance         | Euclidean (L2) distance between 512-D embeddings       |
| Hand-crafted branch distance | Spearman rank distance on HOG features (1 - rho)       |
| Normalization method         | Min-max normalization to `[0, 1]` per distance vector  |
| Fusion method                | Sum of normalized scores (equal weight, 1:1)           |
| Multi-sketch forensic mode   | K=8 augmented variants; minimum distance is taken      |
| 2D Augmentation types        | Horizontal flip, ±10° rotation, brightness/contrast jitter |

---

### A2.8 Ablation Study Configuration

Three ablation experiments are provided to isolate the contribution of each system component:

**1. Single vs. Multi-Sketch Fusion (`evaluate_multisketch.py`):**
```bash
python3 generate_synthetic_sketches.py --k 8
python3 evaluate_multisketch.py
```
Prints Rank-N for single-sketch baseline and multi-sketch fused result side by side, with per-identity breakdown.

**2. Deep-Only vs. HOG-Only vs. Fused (`fusion_eval.py`):**
```bash
python3 fusion_eval.py
```
Prints three separate Rank-N tables to validate whether fusion improves over either individual method.

**3. Standard Gallery vs. Extended Gallery (`evaluate_extended_gallery.py`):**
```bash
python3 extend_gallery.py --n_distractors 500
python3 evaluate_extended_gallery.py
```
Demonstrates how small closed-set gallery evaluations can overstate real-world performance.

---

### A2.9 Execution Commands — System Summary

| Task                                       | Command                                                                       |
|--------------------------------------------|-------------------------------------------------------------------------------|
| Full training pipeline                     | `python3 pipeline.py train`                                                   |
| Single sketch query (full fused pipeline)  | `python3 pipeline.py match --sketch <path> --mode viewed --top_k 5`          |
| Single sketch query (forensic mode)        | `python3 pipeline.py match --sketch <path> --mode forensic --top_k 5`        |
| Evaluate full pipeline (standard gallery)  | `python3 pipeline.py evaluate --mode viewed`                                  |
| Evaluate full pipeline (extended gallery)  | `python3 pipeline.py evaluate --mode viewed --gallery extended_gallery.json`  |
| Deep-only query                            | `python3 predict.py --sketch <path> --top_k 5`                               |
| Ablation: fusion vs. individual branches   | `python3 fusion_eval.py`                                                      |
| Ablation: multi-sketch fusion              | `python3 evaluate_multisketch.py`                                             |
| Ablation: extended gallery                 | `python3 evaluate_extended_gallery.py`                                        |

---

### A2.10 Offline 3DMM Renderer (Advanced Component)

An advanced offline component, `render_3dmm.py`, implements a custom software-based 3D Morphable Model renderer using the **Basel Face Model 2019** (`model2019_bfm.h5`, ~277 MB):

| Component               | Detail                                                        |
|-------------------------|---------------------------------------------------------------|
| Model file              | `model2019_bfm.h5` (Basel Face Model 2019, HDF5 format)      |
| PCA bases               | Shape, Colour, Expression                                      |
| Shape variance scale    | 0.6                                                            |
| Colour variance scale   | 0.5                                                            |
| Expression variance scale | 0.4                                                          |
| 3D projection method    | Orthographic projection                                        |
| Lighting model          | Diffuse directional lighting (surface normals)                 |
| Rendering algorithm     | Painter's algorithm (back-to-front depth sort)                 |
| Rendering backend       | NumPy + Matplotlib Agg (headless, no OpenGL required)          |
| Usage in pipeline       | Offline only; `pipeline.py` uses 2D augmentation by default   |

> **Important:** The 3DMM renderer is an offline research tool. The live `pipeline.py` uses 2D augmentation (flips, rotations, brightness jitter) as a computationally efficient proxy. True 3DMM-based synthetic variants must be generated separately using `render_3dmm.py` and ingested via `import_3dmm_synthetics.py`.

---

---

# CONCLUSION

## Conclusion

This micro-project has successfully designed, implemented, and evaluated a complete Forensic Face Photo-Sketch Recognition System using deep transfer learning and score-level feature fusion. The system addresses the fundamental challenge of the modality gap that exists between forensic sketches — sparse, structurally distorted, hand-drawn images — and high-quality face photographs typically stored in law enforcement databases.

### What Was Developed

A full end-to-end forensic sketch recognition pipeline was implemented in Python using PyTorch. The system encompasses data preparation and pairing, a two-stage fine-tuning regime for a deep facial embedding model, HOG-based structural feature extraction, score-level fusion, and gallery ranking. The system is fully operational via the unified `pipeline.py` entry point and includes additional modules for ablation studies, extended gallery testing, and an offline 3D Morphable Model (3DMM) software renderer.

### How the System Works

The pipeline accepts a forensic sketch as input, preprocesses it to a 160×160 RGB tensor, and passes it through two parallel feature extraction branches. The first branch uses **InceptionResNetV1** — pretrained on VGGFace2 and further fine-tuned on the CUFS dataset in two stages (cross-entropy classification followed by Triplet Margin Loss metric learning) — to produce a **512-dimensional L2-normalized embedding vector**. The second branch extracts an **8,100-dimensional HOG feature descriptor** from a 128×128 grayscale version of the sketch, capturing the local gradient orientation structure of the face. The two distance scores (L2 for deep embeddings and Spearman for HOG) against all gallery photographs are independently min-max normalized to `[0, 1]` and summed to produce a final fused similarity ranking. Gallery photographs are sorted by ascending fused distance, and the Top-K candidate identities are presented as the recognition output.

### Why Combining Deep and Structural Features Is Useful

Deep learning models, when trained on limited data such as the 606 subject pairs in the CUFS dataset, are susceptible to overfitting. A deep embedding alone may produce overconfident similarity scores that do not generalize to unseen probe sketches, especially forensic sketches drawn from memory with significant geometric distortion. HOG features, in contrast, are entirely hand-crafted and parameter-free: they encode the spatial structure of oriented edges without any learned bias. By fusing the global semantic understanding of the deep network with the local geometric precision of HOG features — connected through the rank-invariant Spearman correlation metric — the system creates a complementary safety mechanism. When the neural network makes a poor prediction due to domain mismatch, the HOG branch can partially compensate by independently verifying structural edge alignment. This heterogeneous fusion is a well-established principle in biometric recognition, and the ablation study provided by `fusion_eval.py` allows direct empirical comparison of the three configurations: deep-only, HOG-only, and fused.

### How Top-K Ranking Helps Forensic Investigation

A critical operational insight in forensic recognition is that **Rank-1 accuracy is not always the only meaningful metric**. In a real investigative scenario, a forensic investigator does not require the system to single-handedly identify a suspect with certainty. Rather, the system functions as an intelligent search tool that dramatically narrows the candidate space — for example, from a database of thousands of photographs down to a shortlist of 5 to 10 candidates. A Rank-5 or Rank-10 match is operationally highly valuable: the investigator can then apply domain expertise, witness interviews, and additional corroborating evidence to identify the correct individual from this small shortlist. The system's output format — a ranked list of candidate identity IDs, photo file paths, and numerical fused similarity distances — is explicitly designed to support this investigative workflow.

### Practical Significance

The system demonstrates the feasibility of applying transfer learning from large-scale face recognition datasets (VGGFace2) to the specialized and data-scarce forensic sketch domain. By using a pretrained backbone and freezing its lower layers, the system leverages millions of faces worth of learned facial geometry while only adapting its upper layers to the cross-modal photo-sketch task. This approach is practically significant because it reduces the quantity of labelled photo-sketch training data required — a major bottleneck in forensic AI applications, where acquiring matched photo-sketch datasets is costly and access-restricted.

The extended gallery evaluation (using LFW distractor photographs) further demonstrates that the system maintains its discriminative capability when embedded in a realistically sized database — a requirement that purely closed-set academic benchmarks often fail to assess. It must be clearly stated that this system is designed as an **investigative support tool** to assist forensic professionals in narrowing candidate identities. It must not be used as a standalone final identification authority. All outputs represent probabilistic leads that require further human verification, corroborating evidence, and due investigative process before any action is taken.

### Current Limitations

The system has the following limitations that must be acknowledged:

1. **No automated face detection:** The system assumes all input images are pre-cropped to the face region. A real deployment would require an upstream face detection module (such as MTCNN or RetinaFace) to locate and crop faces from raw surveillance frames or photographs.
2. **Small training dataset:** The CUFS dataset contains only 606 subject pairs. Deep models benefit from millions of training examples, and the limited dataset size constrains the generalization capability of the fine-tuned model.
3. **Viewed sketches only:** The CUFS dataset contains "viewed" sketches — drawn by an artist while looking at the photograph. Real forensic sketches drawn from witness memory contain far greater geometric distortion. The system provides a `--mode forensic` option using 2D augmentation as a proxy, but this cannot fully replicate the structural errors of real forensic drawings.
4. **2D augmentation proxy:** The live pipeline uses 2D geometric augmentation (flips, rotations, brightness/contrast jitter) as a practical substitute for 3DMM-based structural synthesis. The offline 3DMM renderer is available but not integrated into the real-time inference path.
5. **No cross-dataset validation:** The system was evaluated only on CUFS. Evaluation on additional datasets (e.g., PRIP-VSGC) would be required to validate generalization to unseen sketch styles and demographics.
6. **Fixed fusion weights:** The current fusion uses a simple 1:1 sum of normalized scores. A learned weighted combination may provide marginal improvements but risks overfitting on the small dataset.

### Possible Future Improvements

1. **Automated face alignment:** Integrating MTCNN or RetinaFace as a preprocessing step to automatically detect and crop faces from unaligned images, enabling deployment on raw police database photographs.
2. **Differentiable 3D rendering:** Replacing the CPU-based Matplotlib Agg renderer with PyTorch3D to enable GPU-accelerated, differentiable 3D rendering — potentially allowing 3D-aware end-to-end training.
3. **Learned fusion weights:** Replacing the equal-weight `1:1` score fusion with a learned weighted combination (e.g., via a small SVM or logistic regression trained on the validation set) to adaptively balance the deep and structural branches.
4. **Larger and more diverse datasets:** Training on larger forensic sketch datasets or augmenting CUFS with additional annotated cross-modal pairs would substantially improve generalization.
5. **Multi-modal input:** Extending the system to accept natural language witness descriptions alongside the sketch, enabling a joint visual-linguistic retrieval system.
6. **REST API deployment:** Wrapping the inference pipeline as a REST API microservice for integration with existing law enforcement database systems such as CCTNS.

In summary, this project demonstrates that combining deep transfer learning with traditional structural feature extraction and score-level fusion produces a practically viable forensic sketch recognition system. The pipeline is transparent, modular, and reproducible, and it provides a meaningful investigative aid to forensic professionals — narrowing thousands of database candidates to a ranked shortlist of the most probable identities, thereby enhancing the efficiency and effectiveness of forensic identification workflows.

---

---

# REFERENCES

[1] C. Galea and R. A. Farrugia, "Forensic Face Photo-Sketch Recognition Using a Deep Learning-Based Architecture," *IEEE Transactions on Information Forensics and Security (TIFS)*, 2025.

[2] X. Wang and X. Tang, "Face Photo-Sketch Synthesis and Recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 31, no. 11, pp. 1955–1967, Nov. 2009. [CUFS / CUHK Face Sketch Database]

[3] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 815–823.

[4] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman, "VGGFace2: A Dataset for Recognising Faces Across Pose and Age," in *Proceedings of the IEEE International Conference on Automatic Face and Gesture Recognition (FG)*, 2018, pp. 67–74.

[5] N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, vol. 1, 2005, pp. 886–893.

[6] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019. [Online]. Available: https://pytorch.org

[7] T. I. Cannings, "facenet-pytorch: Pretrained PyTorch face detection (MTCNN) and recognition (InceptionResNetV1) models," GitHub Repository, 2020. [Online]. Available: https://github.com/timesler/facenet-pytorch

[8] S. van der Walt, J. L. Schonberger, J. Nunez-Iglesias, F. Boulogne, J. D. Warner, N. Yager, E. Gouillart, and T. Yu, "scikit-image: Image Processing in Python," *PeerJ*, vol. 2, p. e453, 2014. [Online]. Available: https://scikit-image.org [used for HOG feature extraction: `skimage.feature.hog`]

[9] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000. [Online]. Available: https://opencv.org [referenced for general image processing context; Pillow used in this implementation]

[10] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, et al., "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020. [Online]. Available: https://scipy.org [used for Spearman rank correlation: `scipy.stats.spearmanr`]
