"""
import_3dmm_synthetics.py

Bridge script: reads a CSV manifest of 3DMM-rendered images and writes
multisketch_pairs.json in the exact format expected by the rest of the pipeline.

The CSV must have two columns (header row required):
    identity_id,image_path
    0,/path/to/renders/subject0_variant0.png
    0,/path/to/renders/subject0_variant1.png
    1,/path/to/renders/subject1_variant0.png

identity_id must match the numbering in test_pairs.json.

Run:
    python3 import_3dmm_synthetics.py --manifest my_3dmm_renders.csv
    python3 import_3dmm_synthetics.py --manifest my_3dmm_renders.csv --dry_run

Output:
    multisketch_pairs.json  (same schema as generate_synthetic_sketches.py output)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

OUTPUT_FILE = "multisketch_pairs.json"
TEST_PAIRS_FILE = "test_pairs.json"


def load_test_pairs():
    if not os.path.exists(TEST_PAIRS_FILE):
        print(f"ERROR: {TEST_PAIRS_FILE} not found. Run prepare_data.py first.")
        sys.exit(1)
    with open(TEST_PAIRS_FILE) as f:
        return json.load(f)


def load_manifest(csv_path):
    if not os.path.exists(csv_path):
        print(f"ERROR: manifest file not found: {csv_path}")
        sys.exit(1)

    renders_by_id = defaultdict(list)
    errors = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "identity_id" not in reader.fieldnames or "image_path" not in reader.fieldnames:
            print(f"ERROR: CSV must have columns 'identity_id' and 'image_path'. "
                  f"Found: {reader.fieldnames}")
            sys.exit(1)

        for row_num, row in enumerate(reader, start=2):
            try:
                identity_id = int(row["identity_id"].strip())
            except ValueError:
                errors.append(f"  Row {row_num}: identity_id '{row['identity_id']}' is not an integer")
                continue

            image_path = row["image_path"].strip()
            if not os.path.exists(image_path):
                errors.append(f"  Row {row_num}: file not found: {image_path}")
                continue

            renders_by_id[identity_id].append(image_path)

    if errors:
        print(f"Found {len(errors)} error(s) in manifest:")
        for e in errors:
            print(e)
        sys.exit(1)

    return dict(renders_by_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                        help="Path to CSV file with columns: identity_id,image_path")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output JSON file (default: {OUTPUT_FILE})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate the manifest and show what would be imported, but don't write.")
    args = parser.parse_args()

    test_pairs = load_test_pairs()
    test_by_id = {entry["identity_id"]: entry for entry in test_pairs}
    valid_ids = set(test_by_id.keys())
    print(f"Loaded {len(test_pairs)} test identities from {TEST_PAIRS_FILE}.")

    renders_by_id = load_manifest(args.manifest)

    unknown_ids = set(renders_by_id.keys()) - valid_ids
    if unknown_ids:
        print(f"ERROR: {len(unknown_ids)} identity_id(s) in manifest not in {TEST_PAIRS_FILE}:")
        for uid in sorted(unknown_ids):
            print(f"  identity_id={uid}")
        sys.exit(1)

    missing_ids = valid_ids - set(renders_by_id.keys())
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} test identity(s) have no 3DMM renders:")
        for mid in sorted(missing_ids):
            print(f"  identity_id={mid}  (sketch: {test_by_id[mid]['sketch']})")
        print("These identities will be excluded from multisketch evaluation.\n")

    total_variants = sum(len(v) for v in renders_by_id.values())
    covered_ids = sorted(renders_by_id.keys())
    print(f"Manifest OK: {total_variants} render(s) across {len(covered_ids)} identity(s).")
    for iid in covered_ids:
        print(f"  identity_id={iid}: {len(renders_by_id[iid])} variant(s)")

    if args.dry_run:
        print("\n[dry_run] No files written.")
        return

    output_entries = []
    for iid in covered_ids:
        base = test_by_id[iid]
        output_entries.append({
            "identity_id": iid,
            "photo": base["photo"],
            "sketch": base["sketch"],
            "synthetic_sketches": renders_by_id[iid],
        })

    with open(args.output, "w") as f:
        json.dump(output_entries, f, indent=2)

    print(f"\nWrote {args.output} with {len(output_entries)} entries.")
    print("\nNext steps:")
    print(f"  python3 pipeline.py evaluate --mode forensic")
    print(f"  python3 evaluate_multisketch.py")
    print(f"  python3 fusion_eval.py")


if __name__ == "__main__":
    main()
