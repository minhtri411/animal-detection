# ── 2. Imports ───────────────────────────────────────────────────────
import json, random, math
from pathlib import Path
from functools import partial

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# RT-DETR dùng RTDetr* thay vì Detr*
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data/processed/coco_dataset_processed"  
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "rtdetr"
MODEL_CHECKPOINT = "PekingU/rtdetr_r50vd"

CONFIG = dict(
    epochs          = 20,      
    batch_size      = 4,       
    num_workers     = 2,
    lr_backbone     = 1e-5,    
    lr_transformer  = 1e-4,    
    weight_decay    = 1e-4,
    warmup_epochs   = 5,       
    lr_drop_epoch   = 60,      
    lr_drop_gamma   = 0.1,

    max_grad_norm   = 0.1,     

    scale_min       = 480,     
    scale_max       = 800,     

    threshold       = 0.0,     
    vis_threshold   = 0.5,    
    vis_images      = 4,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True

dataset_root = Path(DATASET_ROOT)
output_dir   = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# ── 3. Dataset ───────────────────────────────────────────────────────
class CocoForRTDetr(Dataset):
    """
    COCO-format dataset cho RT-DETR.
    Annotation bbox: [x, y, w, h] (COCO) → processor tự chuyển sang cxcywh.
    """
    def __init__(self, images_dir, annotation_file, processor, cat_id_to_label0):
        self.images_dir       = Path(images_dir)
        self.coco             = COCO(str(annotation_file))
        self.image_ids        = list(self.coco.imgs.keys())
        self.processor        = processor
        self.cat_id_to_label0 = cat_id_to_label0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id   = self.image_ids[idx]
        image_info = self.coco.loadImgs([image_id])[0]
        image      = Image.open(self.images_dir / image_info["file_name"]).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns    = self.coco.loadAnns(ann_ids)

        converted_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            converted_anns.append({
                "image_id"   : image_id,
                "bbox"       : [x, y, w, h],         
                "category_id": self.cat_id_to_label0[ann["category_id"]],
                "area"       : ann.get("area", w * h),
                "iscrowd"    : ann.get("iscrowd", 0),
            })

        target  = {"image_id": image_id, "annotations": converted_anns}
        encoded = self.processor(images=image, annotations=target, return_tensors="pt")
        return encoded["pixel_values"].squeeze(0), encoded["labels"][0]


def collate_fn(batch, processor):
    """Pad ảnh về cùng kích thước trong batch."""
    pixel_values = [item[0] for item in batch]
    labels       = [item[1] for item in batch]
    encoding     = processor.pad(pixel_values, return_tensors="pt")
    # encoding = processor.pad(pixel_values)
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask"  : encoding["pixel_mask"],
        "labels"      : labels,
    }

# ── 4. Class mapping ─────────────────────────────────────────────────
with open(dataset_root / "train" / "_annotations.coco.json", encoding="utf-8") as f:
    train_json = json.load(f)

categories       = sorted(train_json["categories"], key=lambda x: x["id"])
cat_ids          = [c["id"] for c in categories]
cat_id_to_label0 = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
id2label         = {idx: c["name"] for idx, c in enumerate(categories)}
label2id         = {v: k for k, v in id2label.items()}

print(f"Classes ({len(categories)}): {id2label}")

# ── 5. Processor & DataLoaders ───────────────────────────────────────
processor = RTDetrImageProcessor.from_pretrained(MODEL_CHECKPOINT)

train_ds = CocoForRTDetr(
    dataset_root / "train",
    dataset_root / "train" / "_annotations.coco.json",
    processor, cat_id_to_label0)

val_ds = CocoForRTDetr(
    dataset_root / "valid",
    dataset_root / "valid" / "_annotations.coco.json",
    processor, cat_id_to_label0)

test_ds = CocoForRTDetr(
    dataset_root / "test",
    dataset_root / "test"  / "_annotations.coco.json",
    processor, cat_id_to_label0)

print(f"Split — train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

_collate     = partial(collate_fn, processor=processor)
train_loader = DataLoader(
    train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=CONFIG["num_workers"], collate_fn=_collate, pin_memory=True)
val_loader   = DataLoader(
    val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=CONFIG["num_workers"], collate_fn=_collate)
test_loader  = DataLoader(
    test_ds,  batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=CONFIG["num_workers"], collate_fn=_collate)

# ── 6. Model ─────────────────────────────────────────────────────────
model = RTDetrForObjectDetection.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels             = len(categories),
    ignore_mismatched_sizes= True,
    id2label               = id2label,
    label2id               = label2id,
).to(device)

# ── 6a. Param groups: backbone LR nhỏ hơn 10x ───────────────────────
backbone_params     = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
non_backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params,     "lr": CONFIG["lr_backbone"]},
        {"params": non_backbone_params, "lr": CONFIG["lr_transformer"]},
    ],
    weight_decay=CONFIG["weight_decay"],
)

# ── 6b. Scheduler: linear warmup → StepLR ───────────────────────────
warmup_epochs = CONFIG["warmup_epochs"]

def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
step_scheduler   = StepLR(optimizer,
                           step_size=CONFIG["lr_drop_epoch"],
                           gamma=CONFIG["lr_drop_gamma"])
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, step_scheduler],
    milestones=[warmup_epochs])

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params — total: {total_params:,} | trainable: {trainable_params:,}")
print(f"LR     — backbone: {CONFIG['lr_backbone']} | transformer: {CONFIG['lr_transformer']}")
print(f"Model  — {MODEL_CHECKPOINT}")

# ── 7. Train one epoch ───────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, epoch, max_grad_norm):
    model.train()
    total = 0.0

    for step, batch in enumerate(loader, 1):
        pv  = batch["pixel_values"].to(device)
        pm  = batch["pixel_mask"].to(device)
        lbl = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        # ── Forward ─────────────────────────────────────────────────
        # RT-DETR loss = weighted sum(focal_loss + L1_box + GIoU_box)
        outputs = model(pixel_values=pv, pixel_mask=pm, labels=lbl)
        loss    = outputs.loss

        # ── Backward ────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total += loss.item()

        if step % 100 == 0 or step == len(loader):
            # Log loss components nếu có
            loss_info = f"loss={loss.item():.4f}"
            if hasattr(outputs, "loss_dict") and outputs.loss_dict:
                ld = outputs.loss_dict
                parts = []
                for k in ("loss_vfl", "loss_bbox", "loss_giou"):   # RT-DETR keys
                    if k in ld:
                        parts.append(f"{k.replace('loss_', '')}={float(ld[k]):.3f}")
                if parts:
                    loss_info += "  (" + "  ".join(parts) + ")"
            print(f"  Epoch {epoch} | {step:>4}/{len(loader)} | {loss_info}")

    return total / max(len(loader), 1)


# ── 8. Evaluate ──────────────────────────────────────────────────────
def evaluate_rtdetr(model, loader, device, processor, threshold=0.0):
    """
    Evaluate RT-DETR với torchmetrics MeanAveragePrecision.
    RT-DETR trả boxes dưới dạng xyxy (sau post_process).
    GT boxes trong labels là cx cy w h (normalized) → cần chuyển sang xyxy pixel.
    """
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    with torch.no_grad():
        for batch in loader:
            pv     = batch["pixel_values"].to(device)
            pm     = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pv, pixel_mask=pm)

            # RT-DETR post_process trả xyxy pixel
            t_sizes = torch.stack([t["orig_size"] for t in labels])  # (N, 2) h,w
            results = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=t_sizes)

            preds, gts = [], []
            for pred, tgt in zip(results, labels):
                preds.append({
                    "boxes" : pred["boxes"].detach().cpu(),
                    "scores": pred["scores"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu() + 1,   # 1-indexed cho torchmetrics
                })

                # GT: boxes là cxcywh normalized → xyxy pixel
                gb    = tgt["boxes"].detach().cpu()          # (M, 4) cxcywh norm
                h, w  = tgt["orig_size"].cpu().tolist()
                cx, cy, bw, bh = gb[:, 0], gb[:, 1], gb[:, 2], gb[:, 3]
                xyxy  = torch.stack([
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ], dim=1)

                gts.append({
                    "boxes" : xyxy,
                    "labels": tgt["class_labels"].detach().cpu() + 1,
                })

            metric.update(preds, gts)

    return metric.compute()


# ── 9. Training loop ─────────────────────────────────────────────────
def main():
    best_map = -1.0
    history  = []

    print(f"\n🚀 Bắt đầu train RT-DETR — {CONFIG['epochs']} epochs")
    print(f"   Model : {MODEL_CHECKPOINT}")
    print(f"   Device: {device}  |  batch_size={CONFIG['batch_size']}\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_tr = optimizer.param_groups[1]["lr"]
        print(f"\n{'='*60}  Epoch {epoch}/{CONFIG['epochs']}")
        print(f"  lr_backbone={lr_bb:.2e}  |  lr_transformer={lr_tr:.2e}")

        train_loss  = train_one_epoch(
            model, optimizer, train_loader, device, epoch, CONFIG["max_grad_norm"])
        val_metrics = evaluate_rtdetr(
            model, val_loader, device, processor, CONFIG["threshold"])

        scheduler.step()

        row = dict(
            epoch          = epoch,
            train_loss     = float(train_loss),
            val_map        = float(val_metrics["map"]),
            val_map_50     = float(val_metrics["map_50"]),
            val_map_75     = float(val_metrics["map_75"]),
            val_mar_100    = float(val_metrics["mar_100"]),
            lr_backbone    = lr_bb,
            lr_transformer = lr_tr,
        )
        history.append(row)

        print(f"\n  📈 loss={train_loss:.4f} | mAP={row['val_map']:.4f} | "
            f"mAP@50={row['val_map_50']:.4f} | mAP@75={row['val_map_75']:.4f} | "
            f"mAR@100={row['val_mar_100']:.4f}")

        if row["val_map"] > best_map:
            best_map = row["val_map"]
            model.save_pretrained(output_dir / "best_rtdetr")
            processor.save_pretrained(output_dir / "best_rtdetr")
            print(f"  💾 Saved best checkpoint (mAP={best_map:.4f})")

    print("\n✅ Training xong!")

    # ── 10. Test evaluation ──────────────────────────────────────────────
    print("\n🔄 Load best checkpoint & evaluate trên test set...")
    best_model = RTDetrForObjectDetection.from_pretrained(
        output_dir / "best_rtdetr").to(device)
    best_proc  = RTDetrImageProcessor.from_pretrained(output_dir / "best_rtdetr")

    test_metrics = evaluate_rtdetr(
        best_model, test_loader, device, best_proc, CONFIG["threshold"])

    print("\n" + "="*50)
    print("  📊 KẾT QUẢ TẬP TEST — RT-DETR r50vd")
    print("="*50)
    for k in ("map", "map_50", "map_75", "mar_1", "mar_10", "mar_100"):
        print(f"  {k:<12}: {float(test_metrics[k]):.4f}")
    print("="*50)

    # ── 11. Lưu metrics JSON ─────────────────────────────────────────────
    final_results = dict(
        model        = f"RT-DETR ({MODEL_CHECKPOINT})",
        dataset_root = str(dataset_root),
        split        = dict(train=len(train_ds), valid=len(val_ds), test=len(test_ds)),
        classes      = id2label,
        config       = CONFIG,
        best_val_map = best_map,
        test_metrics = {k: float(test_metrics[k])
                        for k in ("map", "map_50", "map_75", "mar_1", "mar_10", "mar_100")},
        history      = history,
    )
    with open(output_dir / "rtdetr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"💾 Metrics → {output_dir}/rtdetr_metrics.json")

    print("\n🎉 Hoàn tất! Files đã lưu tại:", OUTPUT_DIR)
    print(f"   ├── best_rtdetr/         (model checkpoint)")
    print(f"   ├── rtdetr_metrics.json  (metrics + history)")
    print(f"   ├── training_curves.png")
    print(f"   └── predictions.png")


if __name__ == "__main__":
    main()