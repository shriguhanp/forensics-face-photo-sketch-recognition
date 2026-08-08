"""
prepare_data.py
Builds the identity list from matched (photo, sketch) pairs and splits
identities into train/test sets — mirroring the paper's protocol of testing
on subjects never seen during training (Section III-D).

Run: python3 prepare_data.py
Produces: train_pairs.json, test_pairs.json
"""

import json
import random

from pairing import find_pairs

DATA_ROOT = "data/cufs"
TEST_FRACTION = 0.15
SEED = 42


def main():
    random.seed(SEED)

    pairs = find_pairs(DATA_ROOT, verbose=True)
    if not pairs:
        print("No pairs found — run explore_data.py first and fix pairing.py if needed.")
        return

    # Each (photo, sketch) pair IS one identity/class (CUFS has exactly one
    # photo and one sketch per subject).
    identities = [
        {"identity_id": i, "photo": photo, "sketch": sketch}
        for i, (photo, sketch) in enumerate(pairs)
    ]
    random.shuffle(identities)

    n_test = max(1, int(len(identities) * TEST_FRACTION))
    test_identities = identities[:n_test]
    train_identities = identities[n_test:]

    with open("train_pairs.json", "w") as f:
        json.dump(train_identities, f, indent=2)
    with open("test_pairs.json", "w") as f:
        json.dump(test_identities, f, indent=2)

    print(f"Total identities: {len(identities)}")
    print(f"Train identities: {len(train_identities)} -> train_pairs.json")
    print(f"Test identities:  {len(test_identities)} -> test_pairs.json")
    print("\nNo identity appears in both files, matching the paper's protocol "
          "of evaluating on subjects unseen during training.")


if __name__ == "__main__":
    main()
