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

    import os
    import glob
    
    if os.path.isdir(args.gallery):
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')
        gallery_entries = []
        for ext in image_extensions:
            for path in glob.glob(os.path.join(args.gallery, ext)):
                gallery_entries.append({"photo": path, "identity_id": "unknown"})
            for path in glob.glob(os.path.join(args.gallery, ext.upper())):
                gallery_entries.append({"photo": path, "identity_id": "unknown"})
        print(f"Loaded {len(gallery_entries)} images from directory {args.gallery}")
    else:
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
