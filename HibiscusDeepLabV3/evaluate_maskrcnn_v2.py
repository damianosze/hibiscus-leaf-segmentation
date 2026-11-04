
import json
import numpy as np
import torch

from pathlib import Path
from PIL import Image, ImageDraw

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



NUM_CLASSES = 2

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
ANNOTATIONS_FILE = BASE_DIR / "dataset" / "annotations.json"

OUTPUT_DIR = BASE_DIR / "outputs" / "maskrcnn_resnet50_fpn_v2"

MODEL_FILE = OUTPUT_DIR / "best_maskrcnn_v2.pth"
SPLIT_FILE = OUTPUT_DIR / "split_indices.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]



def collate_fn(batch):
    """
    Zachowuje obrazy, targety i nazwy plików
    jako osobne listy zamiast łączyć je w słownik.
    """
    return tuple(zip(*batch))



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

        image = Image.open(
            IMAGES_DIR / info["file_name"]
        ).convert("RGB")

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

            mask_array = np.array(
                mask_image,
                dtype=np.uint8,
                copy=True
            )

            mask = torch.from_numpy(mask_array)

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

        return (
            TF.to_tensor(image),
            target,
            info["file_name"]
        )


def create_model():

    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None
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

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }



def main():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_FILE}"
        )

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono podziału: {SPLIT_FILE}"
        )

    dataset = HibiscusDataset()

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
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
        num_workers=0,
        collate_fn=collate_fn
    )

    model = create_model()

    predictions = []

    print("\nRozpoczynam ewaluację modelu...")

    with torch.no_grad():

        for images, targets, filenames in loader:

            images = [
                image.to(DEVICE)
                for image in images
            ]

            outputs = model(images)

            for output, target, filename in zip(
                outputs, targets, filenames
            ):

                gt_masks = (
                    target["masks"].cpu().numpy().astype(bool)
                )

                if len(gt_masks):
                    gt = np.any(gt_masks, axis=0)
                else:
                    gt = np.zeros(
                        target["masks"].shape[-2:],
                        dtype=bool
                    )

                predictions.append({
                    "file_name": filename,
                    "gt": gt,
                    "masks": (
                        output["masks"]
                        .cpu()
                        .numpy()[:, 0]
                    ),
                    "scores": (
                        output["scores"]
                        .cpu()
                        .numpy()
                    )
                })


    def evaluate(threshold):

        tp = fp = fn = detections = 0

        for sample in predictions:

            gt = sample["gt"]

            pred = np.zeros_like(gt)

            selected = (
                sample["scores"] >= threshold
            )

            detections += int(selected.sum())

            for mask in sample["masks"][selected]:
                pred |= mask >= MASK_THRESHOLD

            tp += int((gt & pred).sum())
            fp += int((~gt & pred).sum())
            fn += int((gt & ~pred).sum())

        return calculate_metrics(tp, fp, fn), detections

    metrics, detections = evaluate(SCORE_THRESHOLD)


    print("\nEwaluacja Mask R-CNN ResNet50 FPN V2")

    print("Liczba zdjęć walidacyjnych:", len(val_dataset))
    print("Próg pewności:", SCORE_THRESHOLD)

    print("\nWyniki zbiorcze:")

    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")

    print("\nPiksele:")

    print("TP:", metrics["tp"])
    print("FP:", metrics["fp"])
    print("FN:", metrics["fn"])

    print("\nLiczba detekcji:", detections)


    print("\nWyniki dla zdjęć:")

    for sample in predictions:

        pred = np.zeros_like(sample["gt"])

        selected = (
            sample["scores"] >= SCORE_THRESHOLD
        )

        for mask in sample["masks"][selected]:
            pred |= mask >= MASK_THRESHOLD

        tp = int((sample["gt"] & pred).sum())
        fp = int((~sample["gt"] & pred).sum())
        fn = int((sample["gt"] & ~pred).sum())

        m = calculate_metrics(tp, fp, fn)

        print(
            f"{sample['file_name']:<35} "
            f"IoU: {m['iou']:.4f} "
            f"F1: {m['f1']:.4f}"
        )


    print("\nWpływ progu pewności:")

    print(
        f"{'Próg':>8} "
        f"{'Detekcje':>10} "
        f"{'IoU':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'F1':>10}"
    )

    threshold_results = {}

    for threshold in THRESHOLDS:

        m, count = evaluate(threshold)

        threshold_results[str(threshold)] = {
            **m,
            "detections": count
        }

        print(
            f"{threshold:>8.2f} "
            f"{count:>10} "
            f"{m['iou']:>10.4f} "
            f"{m['precision']:>12.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f}"
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_DIR / "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump({
            "main_results": metrics,
            "detections": detections,
            "threshold_results": threshold_results
        }, f, indent=2)

    print("\nZapisano wyniki ewaluacji.")


if __name__ == "__main__":
    main()
