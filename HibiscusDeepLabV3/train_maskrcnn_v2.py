from pathlib import Path
import json
import random
import time

import numpy as np
import torch

from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


SEED = 42
EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
NUM_CLASSES = 2

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
ANNOTATIONS_FILE = BASE_DIR / "dataset" / "annotations.json"

OUTPUT_DIR = BASE_DIR / "outputs" / "maskrcnn_resnet50_fpn_v2"

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


class HibiscusDataset(Dataset):

    def __init__(self):

        with open(
            ANNOTATIONS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
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

        image_path = IMAGES_DIR / info["file_name"]

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        masks = []
        boxes = []
        labels = []

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

            mask = torch.from_numpy(
                np.array(mask_image, dtype=np.uint8, copy=True)
            )

            if mask.sum().item() == 0:
                continue

            ys, xs = torch.where(mask > 0)

            boxes.append([
                xs.min().item(),
                ys.min().item(),
                xs.max().item() + 1,
                ys.max().item() + 1
            ])

            masks.append(mask)
            labels.append(1)

        if masks:

            masks = torch.stack(masks)

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64
            )

        else:

            masks = torch.zeros(
                (0, height, width),
                dtype=torch.uint8
            )

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([info["id"]]),
            "area": masks.flatten(1).sum(1).float(),
            "iscrowd": torch.zeros(
                len(labels),
                dtype=torch.int64
            )
        }

        return TF.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model():

    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        NUM_CLASSES
    )

    return model


def calculate_validation_iou(model, loader):

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.inference_mode():

        for images, targets in loader:

            images = [
                image.to(DEVICE)
                for image in images
            ]

            predictions = model(images)

            for prediction, target in zip(
                predictions, targets
            ):

                gt_masks = target["masks"].numpy().astype(bool)

                if len(gt_masks):
                    gt = np.any(gt_masks, axis=0)
                else:
                    gt = np.zeros(
                        target["masks"].shape[-2:],
                        dtype=bool
                    )

                pred = np.zeros_like(gt)

                selected = (
                    prediction["scores"].cpu().numpy() >= 0.5
                )

                predicted_masks = (
                    prediction["masks"].cpu().numpy()
                )

                for mask in predicted_masks[selected]:
                    pred |= mask[0] >= 0.5

                total_tp += int((gt & pred).sum())
                total_fp += int((~gt & pred).sum())
                total_fn += int((gt & ~pred).sum())

    denominator = total_tp + total_fp + total_fn

    return (
        total_tp / denominator
        if denominator else 1.0
    )


def main():

    dataset = HibiscusDataset()

    if len(dataset) < 2:
        raise ValueError("Potrzeba minimum 2 zdjęć.")

    print("Liczba zdjęć:", len(dataset))
    print("Liczba adnotacji:", len(dataset.annotations))

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
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print("Zdjęcia treningowe:", len(train_dataset))
    print("Zdjęcia walidacyjne:", len(val_dataset))

    model = create_model().to(DEVICE)

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005
    )

    best_iou = -1.0
    history = []

    for epoch in range(EPOCHS):

        start = time.time()

        model.train()

        train_loss_sum = 0.0

        for images, targets in train_loader:

            images = [
                image.to(DEVICE)
                for image in images
            ]

            targets = [
                {
                    key: value.to(DEVICE)
                    for key, value in target.items()
                }
                for target in targets
            ]

            optimizer.zero_grad()

            loss_dict = model(images, targets)

            loss = sum(loss_dict.values())

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

        val_iou = calculate_validation_iou(
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
                OUTPUT_DIR / "best_maskrcnn_v2.pth"
            )

            print("Zapisano najlepszy model.")

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "last_maskrcnn_v2.pth"
    )

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(history, f, indent=2)

    print("\nTrening Mask R-CNN FPN V2 zakończony.")
    print("Najlepszy IoU:", best_iou)


if __name__ == "__main__":
    main()