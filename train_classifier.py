"""
train_classifier.py
Stage 1 (mirrors paper Section III-A): fine-tune the pretrained face network
so it recognizes each subject as the same identity regardless of whether the
input is their photo or their sketch. Trained as ordinary classification,
one class per training identity.

Run: python3 train_classifier.py
Produces: stage1_model.pth

CLI options (used by run_ablation.py to sweep configs):
  --augment / --no-augment   toggle training-time augmentation (default: on)
  --epochs N                  number of epochs (default: 15)
  --output PATH               checkpoint filename (default: stage1_model.pth)
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_utils import get_device, load_image_tensor, build_backbone, EMBEDDING_DIM

BATCH_SIZE = 16
LEARNING_RATE = 1e-4


class IdentityDataset(Dataset):
    """Each identity contributes two samples: its photo and its sketch,
    both labeled with the same class index."""

    def __init__(self, pairs_json_path, augment):
        with open(pairs_json_path) as f:
            identities = json.load(f)
        self.samples = []
        for entry in identities:
            label = entry["identity_id"]
            self.samples.append((entry["photo"], label))
            self.samples.append((entry["sketch"], label))
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = load_image_tensor(path, augment=self.augment)
        return tensor, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", default="stage1_model.pth")
    parser.add_argument("--train_pairs", default="train_pairs.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Config: augment={args.augment} epochs={args.epochs} output={args.output}")

    train_dataset = IdentityDataset(args.train_pairs, augment=args.augment)
    num_classes = len({label for _, label in train_dataset.samples})
    print(f"Training identities (classes): {num_classes}")
    print(f"Training samples (photos + sketches): {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    backbone = build_backbone(device)
    classifier_head = nn.Linear(EMBEDDING_DIM, num_classes).to(device)

    params = [p for p in backbone.parameters() if p.requires_grad] + list(classifier_head.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("\nStarting Stage 1 training (classification)...\n")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        classifier_head.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = backbone(images)
            logits = classifier_head(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        print(f"Epoch {epoch:2d}/{args.epochs} | loss={total_loss / total:.4f} | "
              f"train_acc={correct / total:.3f}")

    elapsed = time.time() - start
    print(f"\nStage 1 finished in {elapsed:.1f}s.")

    torch.save({"backbone_state_dict": backbone.state_dict()}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
