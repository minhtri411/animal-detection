import sys
import uuid
import argparse
import gc
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T  
from PIL import Image
from ultralytics import YOLO

# =========================================================
# 1. PARSE COMMAND LINE ARGUMENTS & CONFIG
# =========================================================

APP_TITLE = "🦁 Wildlife Object Detection System"

parser = argparse.ArgumentParser(description=APP_TITLE)
parser.add_argument(
    "--model", 
    type=str, 
    choices=["yolo", "rcnn", "detr"], 
    default="yolo", 
    help="Select model architecture: 'yolo', 'rcnn', or 'detr' (for RT-DETR)"
)
parser.add_argument(
    "--path", 
    type=str, 
    default="runs_raw/object_detection/yolov8n_baseline/weights/best.pt", 
    help="Path to weight file (.pt, .pth) or directory (for RT-DETR)"
)
args = parser.parse_args()

MODEL_MAP = {
    "yolo": "YOLOv8",
    "rcnn": "Faster R-CNN",
    "detr": "RT-DETR"
}

MODEL_TYPE = MODEL_MAP[args.model]
TARGET_MODEL_PATH = Path(args.path)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

COLORS_HEX = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#00C7BE",
    "#30B0C7", "#32ADE6", "#007AFF", "#5856D6", "#AF52DE",
    "#FF2D55", "#A2845E", "#8E8E93", "#636366", "#48484A",
]

def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[0], rgb[1], rgb[2]) 

print("=" * 60)
print(f"[+] CUDA AVAILABLE: {torch.cuda.is_available()}")
print(f"[+] DEVICE: {DEVICE}")
print(f"[+] MODEL ARCHITECTURE: {MODEL_TYPE}")
print(f"[+] DATA PATH: {TARGET_MODEL_PATH}")
print("=" * 60)

# =========================================================
# 2. VALIDATION & LOAD MODEL (DYNAMIC LABELS)
# =========================================================

if not TARGET_MODEL_PATH.exists():
    print(f"\n[ERROR] Target path not found: {TARGET_MODEL_PATH}")
    sys.exit(1)

model = None
processor = None
id2label = {}

if MODEL_TYPE == "YOLOv8":
    print("Loading YOLOv8 architecture...")
    model = YOLO(str(TARGET_MODEL_PATH))
    id2label = model.names

elif MODEL_TYPE == "Faster R-CNN":
    print("Loading Faster R-CNN architecture (MobileNetV3 Backbone)...")
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    _ckpt = torch.load(TARGET_MODEL_PATH, map_location=DEVICE)
    NUM_CLASSES = _ckpt["roi_heads.box_predictor.cls_score.weight"].shape[0]
    print(f"Auto-detected NUM_CLASSES from checkpoint: {NUM_CLASSES}")

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None 
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(_ckpt)
    model.to(DEVICE)
    model.eval()
    
    id2label = {
        0: "Background", 
        1: "objects",
        2: "Elephant", 
        3: "Pig", 
        4: "Tiger", 
        5: "Lion", 
        6: "Wolf"
    }

elif MODEL_TYPE == "RT-DETR":
    if TARGET_MODEL_PATH.is_file():
        print(f"\n[DETR ERROR] The provided path must be a DIRECTORY, not a file!")
        sys.exit(1)
        
    print("Loading RT-DETR (Real-Time DETR) architecture...")
    from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
    
    try:
        processor = RTDetrImageProcessor.from_pretrained(TARGET_MODEL_PATH)
        model = RTDetrForObjectDetection.from_pretrained(TARGET_MODEL_PATH).to(DEVICE)
        model.eval()
        id2label = model.config.id2label 
    except ImportError:
        print("\n[LIBRARY ERROR] The 'timm' library is missing. Run: pip install timm")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to load RT-DETR: {e}")
        sys.exit(1)

print(f"[+] Model loaded successfully! Classes ({len(id2label)}): {id2label}")

# =========================================================
# 3. BUILD DYNAMIC UI HTML HEADER
# =========================================================
classes_html = ""
for idx, (cls_id, cls_name) in enumerate(id2label.items()):
    if str(cls_name).lower() in ["background", "bg", "objects"]:
        continue
    bg_color = COLORS_HEX[idx % len(COLORS_HEX)]
    classes_html += f"<span style='background: {bg_color}; color: #fff; padding: 4px 10px; border-radius: 6px; font-weight: 500; font-size: 14px;'>{cls_id}: {cls_name}</span>&nbsp;\n"

# =========================================================
# 4. CORE LOGIC (INFERENCE)
# =========================================================

def process_image(image_pil, conf_threshold, iou_threshold):
    if image_pil is None or model is None:
        return None, {"predictions": []}

    image_np = np.array(image_pil.convert("RGB"))
    img_drawn = image_np.copy()
    predictions_json = []

    if MODEL_TYPE == "YOLOv8":
        results = model.predict(source=image_np, conf=conf_threshold, iou=iou_threshold, imgsz=640, device=DEVICE, verbose=False, save=False)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = float(box.xywh[0][2]), float(box.xywh[0][3])
                label = id2label.get(cls_id, f"Class {cls_id}")
                
                predictions_json.append({
                    "x": round(x1 + w/2, 1), "y": round(y1 + h/2, 1),
                    "width": round(w, 1), "height": round(h, 1),
                    "confidence": round(conf, 3), "class": label, "class_id": cls_id,
                    "detection_id": str(uuid.uuid4())[:12]
                })

    elif MODEL_TYPE == "Faster R-CNN":
        rcnn_transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
        
        img_tensor = rcnn_transform(image_np).to(DEVICE)
        
        with torch.no_grad():
            outputs = model([img_tensor])[0]
        
        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        
        # Lọc bằng Numpy thay cho NMS bên ngoài
        keep = scores >= conf_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        for box, cls_id, conf in zip(boxes, labels, scores):
            cls_id = int(cls_id)
            label_name = id2label.get(cls_id, f"Class {cls_id}")
            
            if label_name in ["Background", "objects"]:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            predictions_json.append({
                "x": round(x1 + w/2, 1), "y": round(y1 + h/2, 1),
                "width": w, "height": h,
                "confidence": round(float(conf), 3), "class": label_name, "class_id": cls_id,
                "detection_id": str(uuid.uuid4())[:12]
            })

    elif MODEL_TYPE == "RT-DETR":
        image_rgb_pil = Image.fromarray(image_np) 
        
        inputs = processor(images=image_rgb_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([image_rgb_pil.size[::-1]]).to(DEVICE)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]

        boxes = results["boxes"].cpu().tolist()
        labels = results["labels"].cpu().tolist()
        scores = results["scores"].cpu().tolist()

        for score, label_id, box in zip(scores, labels, boxes):
            conf = float(score)
            cls_id = int(label_id)
            
            label_name = id2label.get(cls_id, f"Class {cls_id}")
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            predictions_json.append({
                "x": round(x1 + w/2, 1), "y": round(y1 + h/2, 1),
                "width": w, "height": h,
                "confidence": round(conf, 3), "class": label_name, "class_id": cls_id,
                "detection_id": str(uuid.uuid4())[:12]
            })

    for pred in predictions_json:
        x1 = int(pred["x"] - pred["width"]/2)
        y1 = int(pred["y"] - pred["height"]/2)
        x2 = int(pred["x"] + pred["width"]/2)
        y2 = int(pred["y"] + pred["height"]/2)
        cls_id = pred["class_id"]
        
        color_rgb = hex_to_rgb(COLORS_HEX[cls_id % len(COLORS_HEX)])
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color_rgb, 2)
        
        text = f"{pred['class']} {int(pred['confidence'] * 100)}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        y_text_base = y1 - 5 if y1 > 20 else y1 + th + 5
        cv2.rectangle(img_drawn, (x1, y_text_base - th - 5), (x1 + tw, y_text_base + 5), color_rgb, -1)
        cv2.putText(img_drawn, text, (x1, y_text_base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    gc.collect()

    if not predictions_json:
        out_json = {
            "message": f"Scan complete: No objects detected by {MODEL_TYPE}.",
            "predictions": []
        }
    else:
        out_json = {
            "message": f"Detected {len(predictions_json)} object(s).",
            "predictions": predictions_json
        }

    return Image.fromarray(img_drawn), out_json

# =========================================================
# 5. UI STATE MANAGEMENT & GRADIO
# =========================================================

def update_ui(image_state, conf, iou):
    if image_state is None:
        return None, {"message": "No image uploaded.", "predictions": []}
    return process_image(image_state, conf, iou)

def handle_upload(image_pil):
    if image_pil is None:
        return None, None, {"predictions": []}
    return image_pil, image_pil, {"message": "Image uploaded. Press RUN DETECTION to scan.", "predictions": []}

custom_theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

with gr.Blocks(theme=custom_theme, title=APP_TITLE) as demo:
    
    gr.HTML(
        f"""
        <div style='text-align: center; padding: 10px;'>
            <h1 style='color: #4F46E5; margin-bottom: 5px;'>{APP_TITLE}</h1>
            <p style='font-size: 16px; color: #6B7280; margin-top: 5px;'>
                <b>Supported wildlife classes for detection:</b><br><br>
                <span style='background: #FEE2E2; color: #991B1B; padding: 4px 8px; border-radius: 6px;'>🐘 Elephant</span>&nbsp;
                <span style='background: #FFEDD5; color: #9A3412; padding: 4px 8px; border-radius: 6px;'>🐖 Pig</span>&nbsp;
                <span style='background: #CCFBF1; color: #115E59; padding: 4px 8px; border-radius: 6px;'>🐅 Tiger</span>&nbsp;
                <span style='background: #DCFCE7; color: #166534; padding: 4px 8px; border-radius: 6px;'>🦁 Lion</span>&nbsp;
                <span style='background: #F3E8FF; color: #6B21A8; padding: 4px 8px; border-radius: 6px;'>🐺 Wolf</span>
            </p>
        </div>
        """
    )
    
    original_image_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            main_image = gr.Image(type="pil", label="Workspace", height=600)
        with gr.Column(scale=1):
            gr.Textbox(value=MODEL_TYPE, label="🧠 Active Architecture", interactive=False)
            conf_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.50, step=0.01, label="Confidence Threshold")
            iou_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.50, step=0.01, label="Overlap Threshold (IoU)")
            
            detect_btn = gr.Button("🚀 RUN DETECTION", variant="primary", size="lg")
            json_output = gr.JSON(label="Predictions JSON")

    main_image.upload(fn=handle_upload, inputs=[main_image], outputs=[original_image_state, main_image, json_output])
    detect_btn.click(fn=update_ui, inputs=[original_image_state, conf_slider, iou_slider], outputs=[main_image, json_output])
    main_image.clear(lambda: (None, None, {"predictions": []}), outputs=[original_image_state, main_image, json_output])

if __name__ == "__main__":
    demo.queue() 
    demo.launch(share=True, debug=True)