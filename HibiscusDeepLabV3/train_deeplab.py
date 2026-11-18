import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)


SEED = 42
EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_CLASSES = 2

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
ANNOTATIONS_FILE = BASE_DIR / "dataset" / "annotations.json"
OUTPUT_DIR = BASE_DIR / "outputs" / "deeplabv3_resnet50"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Urządzenie:", DEVICE)


class HibiscusSemanticDataset(Dataset):

    def __init__(self, images_dir, annotation_file):
        self.images_dir = Path(images_dir)

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.annotations_by_image = {}

        for ann in self.annotations:
            self.annotations_by_image.setdefault(
                ann["image_id"], []
            ).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        info = self.images[idx]
        image_path = self.images_dir / info["file_name"]

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        semantic_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for ann in self.annotations_by_image.get(
            info["id"], []
        ):

            segmentation = ann.get("segmentation", [])

            if not isinstance(segmentation, list):
                continue

            mask_image = Image.new(
                "L", (width, height), 0
            )

            draw = ImageDraw.Draw(mask_image)

            for polygon in segmentation:

                if not isinstance(polygon, list):
                    continue

                if len(polygon) < 6 or len(polygon) % 2 != 0:
                    continue

                points = [
                    (polygon[i], polygon[i + 1])
                    for i in range(0, len(polygon), 2)
                ]

                draw.polygon(
                    points,
                    outline=1,
                    fill=1
                )

            mask = np.asarray(mask_image)

            semantic_mask[mask > 0] = 1

        image_tensor = TF.to_tensor(image)

        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        mask_tensor = torch.as_tensor(
            semantic_mask,
            dtype=torch.long
        )

        return image_tensor, mask_tensor


def create_model():

    model = deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.DEFAULT
    )

    in_channels = model.classifier[-1].in_channels

    model.classifier[-1] = nn.Conv2d(
        in_channels,
        NUM_CLASSES,
        kernel_size=1
    )

    if model.aux_classifier is not None:

        aux_channels = model.aux_classifier[-1].in_channels

        model.aux_classifier[-1] = nn.Conv2d(
            aux_channels,
            NUM_CLASSES,
            kernel_size=1
        )

    return model


def calculate_iou(model, loader):

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.inference_mode():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)["out"]

            predictions = outputs.argmax(dim=1)

            pred = predictions == 1
            gt = masks == 1

            total_tp += int((pred & gt).sum().item())
            total_fp += int((pred & ~gt).sum().item())
            total_fn += int((~pred & gt).sum().item())

    denominator = total_tp + total_fp + total_fn

    return (
        total_tp / denominator
        if denominator > 0 else 1.0
    )


def main():

    dataset = HibiscusSemanticDataset(
        IMAGES_DIR,
        ANNOTATIONS_FILE
    )

    if len(dataset) < 2:
        raise ValueError("Potrzeba minimum 2 zdjęć.")

    print("Liczba zdjęć:", len(dataset))

    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)

    val_count = max(1, round(len(indices) * 0.2))

    if val_count >= len(indices):
        raise ValueError("Za mało zdjęć treningowych.")

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    with open(
        OUTPUT_DIR / "split_indices.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump({
            "train": train_indices,
            "val": val_indices
        }, f, indent=2)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Zdjęcia treningowe:", len(train_dataset))
    print("Zdjęcia walidacyjne:", len(val_dataset))

    model = create_model().to(DEVICE)

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    best_iou = -1.0
    history = []

    for epoch in range(EPOCHS):

        start = time.time()

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        train_loss_sum = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = F.cross_entropy(
                outputs["out"],
                masks
            )

            if "aux" in outputs:
                loss += 0.4 * F.cross_entropy(
                    outputs["aux"],
                    masks
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Nieprawidłowa strata treningowa."
                )

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = (
            train_loss_sum / len(train_loader)
        )

        val_iou = calculate_iou(
            model,
            val_loader
        )

        elapsed = time.time() - start

        print(
            f"Epoka {epoch + 1}/{EPOCHS} | "
            f"train loss: {avg_train_loss:.4f} | "
            f"val IoU: {val_iou:.4f} | "
            f"czas: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_iou": val_iou
        })

        if val_iou > best_iou:

            best_iou = val_iou

            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "best_deeplabv3_resnet50.pth"
            )

            print("Zapisano najlepszy model.")

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "last_deeplabv3_resnet50.pth"
    )

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(history, f, indent=2)

    print("\nTrening DeepLabV3 zakończony.")
    print("Najlepszy IoU:", best_iou)


if __name__ == "__main__":
    main()