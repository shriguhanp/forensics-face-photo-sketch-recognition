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
