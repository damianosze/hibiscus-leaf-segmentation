import json
from pathlib import Path

import numpy as np
import torch

from PIL import Image, ImageDraw
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.detection import MeanAveragePrecision


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


MIN_COMPONENT_AREA = 16


def collate_fn(batch):
    return tuple(zip(*batch))


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

        instance_masks = []

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

            mask = np.asarray(mask_image) > 0

            if not mask.any():
                continue

            semantic_mask[mask] = 1
            instance_masks.append(mask)

        if instance_masks:
            instances = np.stack(instance_masks)
        else:
            instances = np.zeros(
                (0, height, width),
                dtype=bool
            )

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

        return (
            image_tensor,
            mask_tensor,
            torch.from_numpy(instances),
            info["file_name"]
        )


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


def extract_instances(foreground, probabilities):


    labeled, count = ndimage.label(foreground)

    masks = []
    scores = []

    for component_id in range(1, count + 1):

        component = labeled == component_id

        if component.sum() < MIN_COMPONENT_AREA:
            continue

        masks.append(component)
        scores.append(float(probabilities[component].mean()))

    height, width = foreground.shape

    if masks:
        masks = np.stack(masks)
    else:
        masks = np.zeros((0, height, width), dtype=bool)

    return masks, np.asarray(scores, dtype=np.float32)


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
        num_workers=0,
        collate_fn=collate_fn
    )

    model = create_model()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    results = []

    map_preds = []
    map_targets = []

    with torch.inference_mode():

        for images, masks, instances, filenames in loader:

            images = torch.stack(images).to(DEVICE)
            masks = torch.stack(masks).to(DEVICE)

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

            # Przybliżone instancje do Mask AP
            probabilities = (
                torch.softmax(outputs, dim=1)[0, 1]
                .cpu()
                .numpy()
            )

            foreground = pred[0].cpu().numpy()

            pred_masks, pred_scores = extract_instances(
                foreground,
                probabilities
            )

            map_preds.append({
                "masks": torch.from_numpy(pred_masks),
                "scores": torch.from_numpy(pred_scores),
                "labels": torch.ones(
                    len(pred_scores), dtype=torch.int64
                )
            })

            gt_instances = instances[0].cpu().numpy().astype(bool)

            map_targets.append({
                "masks": torch.from_numpy(gt_instances),
                "labels": torch.ones(
                    len(gt_instances), dtype=torch.int64
                )
            })

    iou, precision, recall, f1 = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    metric = MeanAveragePrecision(
        iou_type="segm",
        backend="pycocotools"
    )

    metric.update(map_preds, map_targets)

    map_result = metric.compute()

    instance_metrics = {
        "mask_map": float(map_result["map"]),
        "mask_ap50": float(map_result["map_50"]),
        "mask_ap75": float(map_result["map_75"])
    }

    print("\nEwaluacja DeepLabV3-ResNet50")

    print("Liczba zdjęć walidacyjnych:", len(val_dataset))

    print("\nWyniki zbiorcze (piksele):")
    print(f"IoU:       {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nPiksele:")
    print("TP:", total_tp)
    print("FP:", total_fp)
    print("FN:", total_fn)

    print("\nMiary instancyjne (przybliżenie: składowe spójne maski):")
    print(f"Mask mAP:   {instance_metrics['mask_map']:.4f}")
    print(f"Mask AP50:  {instance_metrics['mask_ap50']:.4f}")
    print(f"Mask AP75:  {instance_metrics['mask_ap75']:.4f}")

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
            "instance_metrics": instance_metrics,
            "images": results
        }, f, indent=2)

    print("\nZapisano wyniki ewaluacji.")


if __name__ == "__main__":
    main()