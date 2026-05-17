# Object Detection Application

Ứng dụng nhận dạng đối tượng (Object Detection) sử dụng Deep Learning, được xây dựng trong khuôn khổ đồ án cuối kỳ môn Computer Vision.

Dự án thực hiện:

- Xây dựng bộ dữ liệu object detection
- Huấn luyện và đánh giá 3 hướng tiếp cận khác nhau:
  - CNN-based: Faster R-CNN
  - YOLO-based: YOLOv8
  - Transformer-based: DETR
- So sánh hiệu năng giữa các mô hình
- Triển khai mô hình tốt nhất thành ứng dụng web

---

# Features

- Dataset preprocessing
- Model training
- Model evaluation
- Inference on custom images
- Web deployment
- Performance comparison

---

# Project Structure

```bash
object-detection-app/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── configs/
│
├── notebooks/
│
├── src/
│   ├── dataset.py
│   ├── utils.py
│   │
│   └── models/
│       ├── yolo/
│       ├── cnn/
│       └── transformer/
│
├── outputs/
│
├── weights/
│
├── scripts/
│
├── app/
│
├── docs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Requirements

- Python >= 3.10
- CUDA (optional, recommended)
- Git

---
## Dữ liệu

Do kích thước lớn, bộ dữ liệu không được đưa trực tiếp lên repository.

### Bước 1: Tải dữ liệu

Link tải được lưu tại:

```text
data/raw/link_raw.txt
data/processed/link_processed.txt
```

Mở file tương ứng để lấy link và tải dữ liệu.

---

### Bước 2: Giải nén dữ liệu

Sau khi tải về, giải nén vào đúng thư mục:

```text
data/raw/
data/processed/
```

---

### Cấu trúc sau khi giải nén

```text
data/
├── raw/
└── processed/
```

# Installation

## 1. Clone repository

```bash
git clone <your-repository-url>
cd object-detection-app
```

---

## 2. Create virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

### Linux / MacOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Đặt dữ liệu vào:

```bash
data/raw/
```

Sau đó chạy preprocessing:

notebooks/eda_and_preprocessing.ipynb


Kết quả:

```bash
data/processed/
```
