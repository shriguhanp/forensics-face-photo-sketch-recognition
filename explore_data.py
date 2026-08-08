"""
explore_data.py
Inspect the raw CUFS download and preview the auto-detected photo/sketch
pairing before you trust it for training.

Run: python3 explore_data.py
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

from pairing import find_pairs

DATA_ROOT = "data/cufs"


def print_tree_summary(root):
    print(f"Directory structure under {root}:\n")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        if depth > 3:
            continue
        images = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        indent = "  " * depth
        label = os.path.basename(dirpath) or dirpath
        if images:
            print(f"{indent}{label}/  ({len(images)} images)")
        else:
            print(f"{indent}{label}/")


def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"Could not find {DATA_ROOT}. Run ./download_data.sh first.")
        return

    print_tree_summary(DATA_ROOT)
    print("\nAttempting to auto-pair photos with sketches...\n")
    pairs = find_pairs(DATA_ROOT, verbose=True)
    print(f"\nTotal pairs found: {len(pairs)}")

    if not pairs:
        print("\nNo pairs found. Open pairing.py and adjust PHOTO_KEYWORDS / "
              "SKETCH_KEYWORDS / SKETCH_SUFFIX_PATTERN to match your actual "
              "folder and filename structure shown above.")
        return

    # Preview a handful of pairs so you can visually confirm correctness
    sample = random.sample(pairs, min(6, len(pairs)))
    fig, axes = plt.subplots(2, len(sample), figsize=(3 * len(sample), 6))
    for col, (photo_path, sketch_path) in enumerate(sample):
        axes[0, col].imshow(Image.open(photo_path).convert("RGB"))
        axes[0, col].set_title("photo", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(Image.open(sketch_path).convert("RGB"))
        axes[1, col].set_title("sketch", fontsize=9)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig("sample_pairs.png", dpi=120)
    print("\nSaved sample_pairs.png — open it and confirm each column shows "
          "the SAME person's photo and sketch before proceeding to prepare_data.py.")


if __name__ == "__main__":
    main()
