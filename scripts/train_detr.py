import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DetrForObjectDetection, DetrImageProcessor


class CocoForDetr(Dataset):
    def __init__(self, images_dir, annotation_file, processor, cat_id_to_label0):
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(annotation_file))
        self.image_ids = list(self.coco.imgs.keys())
        self.processor = processor
        self.cat_id_to_label0 = cat_id_to_label0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        converted_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            converted = {
                "image_id": image_id,
                "bbox": [x, y, w, h],
                "category_id": self.cat_id_to_label0[ann["category_id"]],
                "area": ann.get("area", w * h),
                "iscrowd": ann.get("iscrowd", 0),
            }
            converted_anns.append(converted)

        target = {"image_id": image_id, "annotations": converted_anns}
        encoded = self.processor(images=image, annotations=target, return_tensors="pt")

        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"][0]
        return pixel_values, labels


def collate_fn(batch, processor):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }


def evaluate_detr(model, data_loader, device, processor):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            target_sizes = torch.stack([t["orig_size"] for t in labels])
            results = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)

            preds = []
            gts = []
            for pred, tgt in zip(results, labels):
                preds.append(
                    {
                        "boxes": pred["boxes"].detach().cpu(),
                        "scores": pred["scores"].detach().cpu(),
                        "labels": pred["labels"].detach().cpu() + 1,
                    }
                )

                gt_boxes = tgt["boxes"].detach().cpu()
                gt_boxes_xyxy = torch.zeros_like(gt_boxes)
                gt_boxes_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
                gt_boxes_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
                gt_boxes_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
                gt_boxes_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

                h, w = tgt["orig_size"]
                scale = torch.tensor([w, h, w, h], dtype=gt_boxes_xyxy.dtype)
                gt_boxes_xyxy = gt_boxes_xyxy * scale

                gts.append(
                    {
                        "boxes": gt_boxes_xyxy,
                        "labels": (tgt["class_labels"].detach().cpu() + 1),
                    }
                )

            metric.update(preds, gts)

    return metric.compute()


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(data_loader, start=1):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(data_loader)} | Loss: {loss.item():.4f}")

    return total_loss / max(len(data_loader), 1)


def main(args):
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_root / "train" / "_annotations.coco.json", "r", encoding="utf-8") as f:
        train_json = json.load(f)

    categories = sorted(train_json["categories"], key=lambda x: x["id"])
    cat_ids = [c["id"] for c in categories]

    cat_id_to_label0 = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    id2label = {idx: c["name"] for idx, c in enumerate(categories)}
    label2id = {v: k for k, v in id2label.items()}

    print("Class mapping:", id2label)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_ds = CocoForDetr(dataset_root / "train", dataset_root / "train" / "_annotations.coco.json", processor, cat_id_to_label0)
    val_ds = CocoForDetr(dataset_root / "valid", dataset_root / "valid" / "_annotations.coco.json", processor, cat_id_to_label0)
    test_ds = CocoForDetr(dataset_root / "test", dataset_root / "test" / "_annotations.coco.json", processor, cat_id_to_label0)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(categories),
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_map = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_metrics = evaluate_detr(model, val_loader, device, processor)

        epoch_result = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_map": float(val_metrics["map"]),
            "val_map_50": float(val_metrics["map_50"]),
            "val_map_75": float(val_metrics["map_75"]),
            "val_mar_100": float(val_metrics["mar_100"]),
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch}: loss={train_loss:.4f}, "
            f"val mAP={val_metrics['map']:.4f}, "
            f"mAP50={val_metrics['map_50']:.4f}, mAP75={val_metrics['map_75']:.4f}, "
            f"mAR@100={val_metrics['mar_100']:.4f}"
        )

        if val_metrics["map"] > best_map:
            best_map = float(val_metrics["map"])
            model.save_pretrained(output_dir / "best_detr")
            processor.save_pretrained(output_dir / "best_detr")
            print(f"Saved new best checkpoint at epoch {epoch}")

    model = DetrForObjectDetection.from_pretrained(output_dir / "best_detr").to(device)
    test_metrics = evaluate_detr(model, test_loader, device, processor)

    final_results = {
        "model": "DETR (Transformer-based)",
        "dataset_root": str(dataset_root),
        "split": {"train": len(train_ds), "valid": len(val_ds), "test": len(test_ds)},
        "classes": id2label,
        "best_val_map": best_map,
        "test_metrics": {
            "map": float(test_metrics["map"]),
            "map_50": float(test_metrics["map_50"]),
            "map_75": float(test_metrics["map_75"]),
            "mar_1": float(test_metrics["mar_1"]),
            "mar_10": float(test_metrics["mar_10"]),
            "mar_100": float(test_metrics["mar_100"]),
        },
        "history": history,
    }

    with open(output_dir / "detr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("Final test metrics:", final_results["test_metrics"])
    print(f"Saved metrics to: {output_dir / 'detr_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR on COCO-format dataset")
    parser.add_argument("--dataset-root", type=str, default="data/processed/coco_dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/detr")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
