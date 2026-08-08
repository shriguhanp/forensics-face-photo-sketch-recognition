"""
model_utils.py
Shared preprocessing and backbone-loading code used by train_classifier.py,
train_triplet.py, evaluate.py, and predict.py, so all stages use identical
image preprocessing (critical for embeddings to be comparable).
"""

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from facenet_pytorch import InceptionResnetV1

IMAGE_SIZE = 160  # InceptionResnetV1's expected input size
EMBEDDING_DIM = 512


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_image_tensor(path, augment=False):
    """
    Loads an image and preprocesses it exactly the way InceptionResnetV1
    (pretrained on VGGFace2) expects: RGB, resized to 160x160, standardized
    to roughly [-1, 1] via (pixel - 127.5) / 128.
    Works for both photos and (often grayscale) sketches.

    If augment=True, applies light random augmentation (flip, small rotation,
    brightness/contrast jitter) — our stand-in for the paper's 3D Morphable
    Model synthetic image generation (see README section 9). Only use this
    for training data, never for evaluation/prediction.
    """
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    if augment:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, fillcolor=(128, 128, 128))
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    arr = np.array(img).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    tensor = (tensor - 127.5) / 128.0
    return tensor


def build_backbone(device, freeze_until_block=6):
    """
    Loads InceptionResnetV1 pretrained on VGGFace2 and freezes its early
    layers, leaving later blocks trainable — the same "don't relearn edges
    and eyes from scratch" logic as the paper's use of a pretrained VGG-Face.

    freeze_until_block: how many of the early repeated blocks to freeze.
    InceptionResnetV1 exposes named blocks (conv2d_1a ... block8, mixed_7a,
    repeat_3, etc.); we freeze everything up to and including `repeat_1`
    by default and leave the rest trainable.
    """
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)

    trainable_from = ["repeat_2", "mixed_7a", "repeat_3", "block8", "last_linear", "last_bn"]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable_from)

    return model
