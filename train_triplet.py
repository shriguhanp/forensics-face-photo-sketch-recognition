"""
train_triplet.py
Stage 2 (mirrors paper Section III-A, "tuned for verification using triplet
embedding"): fine-tune the Stage 1 model with triplet loss, so that distance
between embeddings directly reflects same/different identity — this is what
retrieval (evaluate.py, predict.py) actually relies on.

Anchor = a sketch, Positive = that identity's photo, Negative = a random
different identity's photo. Same triplet-distance idea as the paper's
verification tuning (Fig. 1, yellow box), minus the paper's use of multiple
synthetic sketches per anchor.

Run: python3 train_triplet.py
Produces: stage2_model.pth

CLI options:
  --augment / --no-augment    toggle training-time augmentation (default: on)
  --epochs N                   number of epochs (default: 15)
  --stage1_model PATH           input Stage 1 checkpoint (default: stage1_model.pth)
  --output PATH                 output checkpoint filename (default: stage2_model.pth)
"""

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_utils import get_device, load_image_tensor, build_backbone

BATCH_SIZE = 16
LEARNING_RATE = 1e-5  # lower than Stage 1 — fine-tuning, not learning from scratch
MARGIN = 0.3


class TripletDataset(Dataset):
    def __init__(self, pairs_json_path, augment):
        with open(pairs_json_path) as f:
            self.identities = json.load(f)
        self.augment = augment

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, idx):
        anchor_entry = self.identities[idx]
        negative_entry = random.choice([e for e in self.identities if e["identity_id"] != anchor_entry["identity_id"]])

        anchor = load_image_tensor(anchor_entry["sketch"], augment=self.augment)
        positive = load_image_tensor(anchor_entry["photo"], augment=self.augment)
        negative = load_image_tensor(negative_entry["photo"], augment=self.augment)
        return anchor, positive, negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--stage1_model", default="stage1_model.pth")
    parser.add_argument("--output", default="stage2_model.pth")
    parser.add_argument("--train_pairs", default="train_pairs.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Config: augment={args.augment} epochs={args.epochs} "
          f"stage1_model={args.stage1_model} output={args.output}")

    train_dataset = TripletDataset(args.train_pairs, augment=args.augment)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training identities: {len(train_dataset)}")

    backbone = build_backbone(device)
    checkpoint = torch.load(args.stage1_model, map_location=device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])

    params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    triplet_loss_fn = nn.TripletMarginLoss(margin=MARGIN)

    print("\nStarting Stage 2 training (triplet embedding)...\n")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        total_loss, n_batches = 0.0, 0

        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            emb_a = F.normalize(backbone(anchor), dim=1)
            emb_p = F.normalize(backbone(positive), dim=1)
            emb_n = F.normalize(backbone(negative), dim=1)

            loss = triplet_loss_fn(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch:2d}/{args.epochs} | triplet_loss={total_loss / n_batches:.4f}")

    elapsed = time.time() - start
    print(f"\nStage 2 finished in {elapsed:.1f}s.")

    torch.save({"backbone_state_dict": backbone.state_dict()}, args.output)
    print(f"Saved {args.output} — this is your final embedding model.")


if __name__ == "__main__":
    main()
