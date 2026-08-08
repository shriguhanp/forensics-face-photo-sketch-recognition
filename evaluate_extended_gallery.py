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
