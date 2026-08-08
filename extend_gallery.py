"""
extend_gallery.py

Substitute for the paper's extended-gallery simulation (Section III-D),
which pads the test gallery with photos of thousands of extra subjects
from MEDS-II, FRGC v2.0, Multi-PIE, and FEI — datasets that require
individual registration/licensing. We use LFW (Labeled Faces in the Wild)
instead: a long-standing, freely available academic face dataset, to add
"distractor" photos that dilute the gallery the same way — making retrieval
meaningfully harder and closer to a real mugshot-database scale.

These distractor photos have NO matching sketch (obviously — nobody drew a
forensic sketch of a randomly chosen LFW subject). They only ever appear as
gallery/distractor entries, never as probes, exactly mirroring how the
paper's extension subjects work.

Run: python3 extend_gallery.py --n_distractors 500
Produces:
  distractor_photos/<n>.jpg
  extended_gallery.json   (test set photos + distractor photos, with a
                            "has_sketch" flag distinguishing the two)
"""

import argparse
import json
import os

from PIL import Image


def fetch_via_sklearn(n_needed):
    """Primary path: scikit-learn's built-in LFW fetcher, which downloads
    from a stable, sklearn-maintained mirror. Requires internet access."""
    from sklearn.datasets import fetch_lfw_people

    print("Downloading LFW via scikit-learn (this can take a few minutes "
          "the first time)...")
    lfw = fetch_lfw_people(min_faces_per_person=1, resize=1.0, color=True)
    images = lfw.images  # [n_samples, h, w, 3], float in [0, 255]
    print(f"LFW dataset loaded: {len(images)} images available.")

    if len(images) < n_needed:
        print(f"WARNING: LFW only has {len(images)} images, fewer than the "
              f"{n_needed} requested — using all available.")
        n_needed = len(images)

    # Sample evenly across the dataset for identity diversity rather than
    # just taking the first N (which tend to cluster on a few celebrities
    # with many photos each in LFW).
    step = max(1, len(images) // n_needed)
    selected_indices = list(range(0, len(images), step))[:n_needed]
    return [images[i] for i in selected_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_distractors", type=int, default=500,
                         help="Number of LFW distractor photos to add to the gallery")
    parser.add_argument("--test_pairs", default="test_pairs.json")
    parser.add_argument("--output_dir", default="distractor_photos")
    parser.add_argument("--lfw_dir", default=None,
                         help="If sklearn's downloader fails (no internet), manually "
                              "download http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz, "
                              "extract it, and point this at the extracted folder instead.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.test_pairs) as f:
        test_identities = json.load(f)

    distractor_paths = []

    if args.lfw_dir:
        print(f"Using manually-downloaded LFW at {args.lfw_dir}")
        found = []
        for root, _dirs, files in os.walk(args.lfw_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    found.append(os.path.join(root, fname))
                    if len(found) >= args.n_distractors:
                        break
            if len(found) >= args.n_distractors:
                break
        for i, src_path in enumerate(found):
            dst_path = os.path.join(args.output_dir, f"{i}.jpg")
            Image.open(src_path).convert("RGB").save(dst_path)
            distractor_paths.append(dst_path)
    else:
        try:
            images = fetch_via_sklearn(args.n_distractors)
            for i, arr in enumerate(images):
                img = Image.fromarray(arr.astype("uint8"))
                dst_path = os.path.join(args.output_dir, f"{i}.jpg")
                img.save(dst_path)
                distractor_paths.append(dst_path)
        except Exception as e:
            print(f"\nERROR downloading LFW via scikit-learn: {e}")
            print("If this is a network issue, manually download "
                  "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz, extract it, "
                  "and rerun with --lfw_dir /path/to/extracted/lfw-deepfunneled")
            return

    # Build the extended gallery: test-set photos (have a matching sketch,
    # used as both probe target and gallery entry) + distractor photos
    # (gallery-only, no sketch).
    extended_gallery = []
    for entry in test_identities:
        extended_gallery.append({
            "identity_id": entry["identity_id"],
            "photo": entry["photo"],
            "has_sketch": True,
        })
    for i, path in enumerate(distractor_paths):
        extended_gallery.append({
            "identity_id": f"distractor_{i}",
            "photo": path,
            "has_sketch": False,
        })

    with open("extended_gallery.json", "w") as f:
        json.dump(extended_gallery, f, indent=2)

    print(f"\nExtended gallery built: {len(test_identities)} real test-set photos + "
          f"{len(distractor_paths)} LFW distractors = {len(extended_gallery)} total.")
    print(f"That's a {len(extended_gallery) / max(len(test_identities), 1):.1f}x larger "
          f"gallery than the plain test set — a harder, more realistic retrieval task.")
    print("Saved extended_gallery.json")


if __name__ == "__main__":
    main()
