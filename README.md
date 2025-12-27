# YOLOv7 Fast

YOLOv7 訓練專案，支援多解析度 COCO 資料集。

Based on [Official YOLOv7](https://github.com/WongKinYiu/yolov7)

## Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training

### Training Parameters

| Parameter | Description | Example |
| :-- | :-- | :-- |
| `--data` | Dataset config (path, classes) | `data/coco320.yaml` |
| `--cfg` | Model architecture (layers, channels) | `cfg/training/yolov7-tiny.yaml` |
| `--img` | Input image size | `320`, `640` |
| `--weights` | Pretrained weights (optional) | `yolov7-tiny.pt` |
| `--batch-size` | Batch size | `32` |
| `--epochs` | Number of epochs | `100` |

### Multi-Resolution COCO Datasets

This project supports pre-resized COCO datasets for faster training:

| Config File | Resolution | Dataset Path |
| :-- | :-: | :-- |
| `data/coco.yaml` | Original | `./coco/` |
| `data/coco320.yaml` | 320x320 | `./coco320/` |
| `data/coco480.yaml` | 480x480 | `./coco480/` |
| `data/coco640.yaml` | 640x640 | `./coco640/` |

### Training Examples

```bash
# Train YOLOv7-Tiny with 320x320 dataset (fastest)
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100

# Train YOLOv7 with 640x640 dataset (standard)
python train.py --data data/coco640.yaml --img 640 --cfg cfg/training/yolov7.yaml --batch-size 32 --epochs 100

# Train with pretrained weights
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --weights yolov7-tiny.pt --batch-size 64
```

### Model Architectures

| Config | Model | Size | Speed |
| :-- | :-- | :-: | :-: |
| `yolov7-tiny.yaml` | YOLOv7-Tiny | Smallest | Fastest |
| `yolov7.yaml` | YOLOv7 | Standard | Fast |
| `yolov7x.yaml` | YOLOv7-X | Large | Medium |
| `yolov7-w6.yaml` | YOLOv7-W6 | X-Large | Slow |
| `yolov7-e6.yaml` | YOLOv7-E6 | XX-Large | Slowest |

## Inference

```bash
# On image
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source image.jpg

# On video
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source video.mp4
```

## Testing

```bash
python test.py --data data/coco320.yaml --img 320 --batch 32 --weights runs/train/exp/weights/best.pt
```

## Project Structure

```
Yolov7fast/
├── cfg/training/          # Model architecture configs
├── data/                  # Dataset configs
│   ├── coco.yaml         # Original COCO
│   ├── coco320.yaml      # 320x320 COCO
│   ├── coco480.yaml      # 480x480 COCO
│   └── coco640.yaml      # 640x640 COCO
├── coco320/              # 320x320 dataset (local)
├── coco480/              # 480x480 dataset
├── coco640/              # 640x640 dataset
└── coco/                 # Original COCO dataset
```
