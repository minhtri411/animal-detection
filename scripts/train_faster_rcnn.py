import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T


class COCODetectionDataset(Dataset):
    def __init__(self, images_dir, annotation_file, cat_id_to_label, transforms=None):
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(annotation_file))
        self.image_ids = list(self.coco.imgs.keys())
        self.cat_id_to_label = cat_id_to_label
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    # ✅ Đổi từ ResNet50 → MobileNetV3: nhẹ hơn ~8x, nhanh hơn ~3x trên CPU
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate_map(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = []
            gts = []
            for out, tgt in zip(outputs, targets):
                preds.append({
                    "boxes": out["boxes"].detach().cpu(),
                    "scores": out["scores"].detach().cpu(),
                    "labels": out["labels"].detach().cpu(),
                })
                gts.append({
                    "boxes": tgt["boxes"].detach().cpu(),
                    "labels": tgt["labels"].detach().cpu(),
                })

            metric.update(preds, gts)

    return metric.compute()


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(data_loader)} | Loss: {losses.item():.4f}")

    return running_loss / max(len(data_loader), 1)


def main(args):
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_root / "train" / "_annotations.coco.json", "r", encoding="utf-8") as f:
        train_json = json.load(f)

    categories = sorted(train_json["categories"], key=lambda x: x["id"])
    cat_ids = [c["id"] for c in categories]

    cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
    id_to_name = {idx + 1: c["name"] for idx, c in enumerate(categories)}

    print("Class mapping:", id_to_name)

    # ✅ Fix warning: dùng v2.Compose thay ToTensor()
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    train_ds = COCODetectionDataset(
        dataset_root / "train", dataset_root / "train" / "_annotations.coco.json",
        cat_id_to_label, transforms=transform
    )
    val_ds = COCODetectionDataset(
        dataset_root / "valid", dataset_root / "valid" / "_annotations.coco.json",
        cat_id_to_label, transforms=transform
    )
    test_ds = COCODetectionDataset(
        dataset_root / "test", dataset_root / "test" / "_annotations.coco.json",
        cat_id_to_label, transforms=transform
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    num_classes = len(categories) + 1
    model = build_model(num_classes).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_map = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_metrics = evaluate_map(model, val_loader, device)
        scheduler.step()

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
            torch.save(model.state_dict(), output_dir / "best_faster_rcnn.pth")
            print(f"Saved new best checkpoint at epoch {epoch}")

    model.load_state_dict(torch.load(output_dir / "best_faster_rcnn.pth", map_location=device))
    test_metrics = evaluate_map(model, test_loader, device)

    final_results = {
        "model": "Faster R-CNN (MobileNetV3 + FPN)",
        "dataset_root": str(dataset_root),
        "split": {"train": len(train_ds), "valid": len(val_ds), "test": len(test_ds)},
        "classes": id_to_name,
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

    with open(output_dir / "faster_rcnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("Final test metrics:", final_results["test_metrics"])
    print(f"Saved metrics to: {output_dir / 'faster_rcnn_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN (MobileNetV3) on COCO-format dataset")
    parser.add_argument("--dataset-root", type=str, default="data/processed/coco_dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/faster_rcnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)