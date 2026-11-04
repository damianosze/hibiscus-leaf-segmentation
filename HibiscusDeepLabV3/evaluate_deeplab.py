
import json
import random
from pathlib import Path

import numpy as np
import torch

from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50


SEED = 42
NUM_CLASSES = 2

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
ANNOTATIONS_FILE = BASE_DIR / "dataset" / "annotations.json"

OUTPUT_DIR = BASE_DIR / "outputs" / "deeplabv3_resnet50"

MODEL_FILE = OUTPUT_DIR / "best_deeplabv3_resnet50.pth"
SPLIT_FILE = OUTPUT_DIR / "split_indices.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class HibiscusSemanticDataset(Dataset):

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

        image = Image.open(
            IMAGES_DIR / info["file_name"]
        ).convert("RGB")

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

                if len(polygon) < 6 or len(polygon) % 2:
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

            semantic_mask[
                np.asarray(mask_image) > 0
            ] = 1

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

        return image_tensor, mask_tensor, info["file_name"]


def create_model():

    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES,
        aux_loss=True
    )

    state_dict = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def calculate_metrics(tp, fp, fn):

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return iou, precision, recall, f1


def main():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    dataset = HibiscusSemanticDataset()

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)

    val_indices = split["val"]

    val_dataset = Subset(
        dataset,
        val_indices
    )

    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = create_model()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    results = []

    with torch.no_grad():

        for images, masks, filenames in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)["out"]

            predictions = outputs.argmax(dim=1)

            pred = predictions == 1
            gt = masks == 1

            tp = int((pred & gt).sum().item())
            fp = int((pred & ~gt).sum().item())
            fn = int((~pred & gt).sum().item())

            total_tp += tp
            total_fp += fp
            total_fn += fn

            iou, precision, recall, f1 = calculate_metrics(
                tp, fp, fn
            )

            results.append({
                "file_name": filenames[0],
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    iou, precision, recall, f1 = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    print("\nEwaluacja DeepLabV3-ResNet50")

    print("Liczba zdjęć walidacyjnych:", len(val_dataset))

    print("\nWyniki zbiorcze:")
    print(f"IoU:       {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nPiksele:")
    print("TP:", total_tp)
    print("FP:", total_fp)
    print("FN:", total_fn)

    print("\nWyniki dla zdjęć:")

    for result in results:

        print(
            f"{result['file_name']:<35} "
            f"IoU: {result['iou']:.4f} "
            f"F1: {result['f1']:.4f}"
        )

    with open(
        OUTPUT_DIR / "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump({
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "images": results
        }, f, indent=2)

    print("\nZapisano wyniki ewaluacji.")


if __name__ == "__main__":
    main()
