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
