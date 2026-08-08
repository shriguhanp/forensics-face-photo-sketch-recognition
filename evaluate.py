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
