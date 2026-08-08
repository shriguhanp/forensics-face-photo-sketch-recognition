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
