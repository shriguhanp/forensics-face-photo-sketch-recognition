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
