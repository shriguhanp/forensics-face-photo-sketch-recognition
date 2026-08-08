"""
pairing.py
Shared logic for finding (photo, sketch) pairs inside the CUFS dataset
folder, whatever its exact internal layout turns out to be (Kaggle mirrors
of research datasets are not always organized consistently).

PAIRING STRATEGY
-----------------
1. Walk the dataset root and classify every image as "photo" or "sketch"
   based on keywords in its path (folder name, then filename as a
   fallback).
2. Group images by their containing "subset" directory (e.g. CUHK / AR /
   XM2VTS each have their own photo+sketch subfolders), so we never
   accidentally pair a photo from one subset with a sketch from another.
3. Within each subset, try to match a photo to its sketch by stripping
   common sketch-only suffixes/prefixes ("-sz1", "_sketch", etc.) from the
   filename and matching the remaining key.
4. If that yields no matches (unfamiliar naming convention) but the photo
   and sketch counts are equal, fall back to pairing by sorted order —
   true for CUFS-style datasets where sorting preserves subject order.

If pairing looks wrong for your download, adjust SKETCH_SUFFIX_PATTERN or
PHOTO_KEYWORDS / SKETCH_KEYWORDS below based on what explore_data.py prints.
"""

import os
import re

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PHOTO_KEYWORDS = ("photo", "photos", "images")
SKETCH_KEYWORDS = ("sketch", "sketches", "drawing")
SKETCH_SUFFIX_PATTERN = re.compile(r"(-sz\d+|_sketch|-sketch|sketch)$", re.IGNORECASE)


def classify_image(path):
    """Return 'photo' or 'sketch' based on path keywords, or None if unclear."""
    lower_path = path.lower()
    for kw in SKETCH_KEYWORDS:
        if kw in lower_path:
            return "sketch"
    for kw in PHOTO_KEYWORDS:
        if kw in lower_path:
            return "photo"
    return None


def find_all_images(root):
    """Return list of (full_path, subset_dir) for every image under root."""
    images = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(dirpath, fname))
    return images


def base_key(filename):
    stem = os.path.splitext(filename)[0]
    stem = SKETCH_SUFFIX_PATTERN.sub("", stem)
    return stem.lower()


def find_pairs(root, verbose=True):
    """
    Returns a list of (photo_path, sketch_path) tuples found under root.
    """
    all_images = find_all_images(root)

    photos, sketches = [], []
    unclassified = []
    for path in all_images:
        cls = classify_image(path)
        if cls == "photo":
            photos.append(path)
        elif cls == "sketch":
            sketches.append(path)
        else:
            unclassified.append(path)

    if verbose:
        print(f"Found {len(photos)} photo-classified, {len(sketches)} "
              f"sketch-classified, {len(unclassified)} unclassified images.")

    # Strategy: group by IMMEDIATE parent folder.
    # For each grandparent, find all photo-folders and sketch-folders under it,
    # then try to pair each photo-folder with each sketch-folder (cross product).
    # This avoids mixing images across multiple sketch subfolders (sketch/, sketches/,
    # cropped_sketch/, original_sketch/) that share the same grandparent.

    from collections import defaultdict

    def parent_dir(path):
        return os.path.dirname(path)

    def grandparent_dir(path):
        return os.path.dirname(os.path.dirname(path))

    # Map each folder to its images
    photo_by_folder  = defaultdict(list)
    sketch_by_folder = defaultdict(list)
    for p in photos:
        photo_by_folder[parent_dir(p)].append(p)
    for s in sketches:
        sketch_by_folder[parent_dir(s)].append(s)

    # Group folders by grandparent so we only pair within the same subtree
    gp_photo_folders  = defaultdict(list)
    gp_sketch_folders = defaultdict(list)
    for folder in photo_by_folder:
        gp_photo_folders[grandparent_dir(folder)].append(folder)
    for folder in sketch_by_folder:
        gp_sketch_folders[grandparent_dir(folder)].append(folder)

    grandparents = sorted(set(gp_photo_folders) | set(gp_sketch_folders))

    pairs = []
    for gp in grandparents:
        pf_list = gp_photo_folders.get(gp, [])
        sf_list = gp_sketch_folders.get(gp, [])
        if not pf_list or not sf_list:
            continue

        # Try every (photo_folder, sketch_folder) combination and collect key matches
        best_pairs = []
        for pf in pf_list:
            for sf in sf_list:
                gp_p = photo_by_folder[pf]
                gp_s = sketch_by_folder[sf]

                photo_by_key  = {base_key(os.path.basename(p)): p for p in gp_p}
                sketch_by_key = {base_key(os.path.basename(s)): s for s in gp_s}
                common_keys   = set(photo_by_key) & set(sketch_by_key)

                if len(common_keys) >= min(len(gp_p), len(gp_s)) * 0.5:
                    folder_pairs = [(photo_by_key[k], sketch_by_key[k]) for k in sorted(common_keys)]
                    if verbose:
                        pf_name = os.path.basename(pf)
                        sf_name = os.path.basename(sf)
                        print(f"  {gp} [{pf_name} ⇔ {sf_name}]: "
                              f"matched {len(common_keys)} pairs by filename key.")
                    best_pairs.extend(folder_pairs)

        if best_pairs:
            pairs.extend(best_pairs)
        else:
            # Fallback: if exactly one photo-folder and one sketch-folder with equal count
            if len(pf_list) == 1 and len(sf_list) == 1:
                gp_p = photo_by_folder[pf_list[0]]
                gp_s = sketch_by_folder[sf_list[0]]
                if len(gp_p) == len(gp_s):
                    for p, s in zip(sorted(gp_p), sorted(gp_s)):
                        pairs.append((p, s))
                    if verbose:
                        print(f"  {gp}: filename matching failed, paired "
                              f"{len(gp_p)} by sorted order (verify with sample_pairs.png).")
                else:
                    if verbose:
                        print(f"  {gp}: could not pair — {len(gp_p)} photos vs "
                              f"{len(gp_s)} sketches, and no filename-key overlap.")
            else:
                if verbose:
                    print(f"  {gp}: no key-matched pairs found across "
                          f"{len(pf_list)} photo-folder(s) and {len(sf_list)} sketch-folder(s).")

    return pairs
