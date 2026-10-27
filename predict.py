
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


SEED = 42
NUM_CLASSES = 2
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
ANNOTATIONS_FILE = BASE_DIR / "dataset" / "annotations.json"
MODEL_FILE = BASE_DIR / "outputs" / "best_maskrcnn_resnet50.pth"

DEVICE = torch.device("cpu")


def create_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, NUM_CLASSES
    )

    state_dict = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def create_ground_truth(info, annotations):
    image_path = IMAGES_DIR / info["file_name"]

    with Image.open(image_path) as img:
        width, height = img.size

    gt = np.zeros((height, width), dtype=bool)
    instance_count = 0

    for ann in annotations:
        if ann["image_id"] != info["id"]:
            continue

        segmentation = ann.get("segmentation", [])

        if not isinstance(segmentation, list):
            continue

        mask_image = Image.new("L", (width, height), 0)
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

            draw.polygon(points, outline=1, fill=1)

        mask = np.asarray(mask_image) > 0

        if mask.any():
            gt |= mask
            instance_count += 1

    return gt, instance_count


def calculate_metrics(gt, pred):
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())

    return calculate_metrics_from_counts(tp, fp, fn)


def calculate_metrics_from_counts(tp, fp, fn):
    denominator = tp + fp + fn
    iou = tp / denominator if denominator else 1.0

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_FILE}")

    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {ANNOTATIONS_FILE}"
        )

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    if len(images) < 2:
        raise ValueError("Zbiór musi zawierać co najmniej 2 zdjęcia.")

    indices = list(range(len(images)))
    random.seed(SEED)
    random.shuffle(indices)

    val_count = max(1, round(len(indices) * 0.2))
    val_indices = indices[:val_count]

    model = create_model()

    results = []
    predictions = []

    for idx in val_indices:
        info = images[idx]
        image_path = IMAGES_DIR / info["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(f"Nie znaleziono zdjęcia: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image_tensor = TF.to_tensor(image).to(DEVICE)

        gt, annotation_count = create_ground_truth(info, annotations)

        with torch.no_grad():
            prediction = model([image_tensor])[0]

        masks = prediction["masks"].cpu().numpy()[:, 0]
        scores = prediction["scores"].cpu().numpy()

        predictions.append({
            "info": info,
            "gt": gt,
            "masks": masks,
            "scores": scores,
            "annotations": annotation_count,
        })

    for sample in predictions:
        gt = sample["gt"]
        masks = sample["masks"]
        scores = sample["scores"]

        selected = scores >= SCORE_THRESHOLD
        pred_union = np.zeros(gt.shape, dtype=bool)

        for mask in masks[selected]:
            pred_union |= mask >= MASK_THRESHOLD

        metrics = calculate_metrics(gt, pred_union)

        results.append({
            "file_name": sample["info"]["file_name"],
            "annotations": sample["annotations"],
            "detections": int(selected.sum()),
            **metrics,
        })

    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)

    aggregate = calculate_metrics_from_counts(
        total_tp, total_fp, total_fn
    )

    all_scores = [
        float(score)
        for sample in predictions
        for score in sample["scores"]
        if score >= SCORE_THRESHOLD
    ]

    print("\nWyniki ewaluacji Mask R-CNN")
    print(f"Liczba zdjęć:       {len(images)}")
    print(f"Zdjęcia walidacyjne: {len(val_indices)}")
    print(f"Liczba adnotacji:   {len(annotations)}")
    print(f"Próg pewności:      {SCORE_THRESHOLD:.1f}")

    print("\nWyniki zbiorcze:")
    print(f"IoU:       {aggregate['iou']:.4f}")
    print(f"Precision: {aggregate['precision']:.4f}")
    print(f"Recall:    {aggregate['recall']:.4f}")
    print(f"F1-score:  {aggregate['f1']:.4f}")

    print("\nPiksele:")
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")

    print("\nDetekcje:")
    print(f"Liczba wykryć: {len(all_scores)}")

    if all_scores:
        print(f"Średnia pewność: {np.mean(all_scores):.4f}")
        print(f"Min. pewność:    {np.min(all_scores):.4f}")
        print(f"Maks. pewność:   {np.max(all_scores):.4f}")

    print("\nWyniki dla zdjęć:")
    print(
        f"{'Zdjęcie':<42} {'GT':>4} {'Det.':>5} "
        f"{'IoU':>7} {'F1':>7}"
    )

    for r in results:
        print(
            f"{r['file_name'][:41]:<42} "
            f"{r['annotations']:>4} "
            f"{r['detections']:>5} "
            f"{r['iou']:>7.4f} "
            f"{r['f1']:>7.4f}"
        )

    print("\nWpływ progu pewności:")
    print(
        f"{'Próg':>6} {'Detekcje':>10} "
        f"{'IoU':>9} {'Precision':>11} "
        f"{'Recall':>9} {'F1':>9}"
    )

    for threshold in THRESHOLDS:
        tp = fp = fn = detections = 0

        for sample in predictions:
            gt = sample["gt"]
            pred_union = np.zeros(gt.shape, dtype=bool)

            selected = sample["scores"] >= threshold
            detections += int(selected.sum())

            for mask in sample["masks"][selected]:
                pred_union |= mask >= MASK_THRESHOLD

            m = calculate_metrics(gt, pred_union)

            tp += m["tp"]
            fp += m["fp"]
            fn += m["fn"]

        m = calculate_metrics_from_counts(tp, fp, fn)

        print(
            f"{threshold:>6.1f} "
            f"{detections:>10} "
            f"{m['iou']:>9.4f} "
            f"{m['precision']:>11.4f} "
            f"{m['recall']:>9.4f} "
            f"{m['f1']:>9.4f}"
        )



if __name__ == "__main__":
    main()
