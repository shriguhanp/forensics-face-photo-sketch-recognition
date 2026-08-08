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
