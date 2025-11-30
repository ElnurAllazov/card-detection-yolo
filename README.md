# 🃏 Playing Card Detection using YOLOv8 🃏

This project detects and classifies playing cards in images using the YOLOv8 object detection model.  
It includes model training, evaluation, and prediction features.

## 💡 Features
- Detect different playing card types in images
- Custom trained dataset
- Training and prediction scripts included
- Achieved 85% detection accuracy

## 🛠  Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV

## How to run 

```
python3 train.py
```

YOLO will automatically create a folder named project, and inside it another folder for the training experiment named Card-Model:

```
project/
└── Card-Model/
    ├── weights/
    │   ├── best.pt        # Best performing model
    │   └── last.pt        # Last training checkpoint
    ├── results.csv        # Training metrics
    ├── confusion_matrix.png
    ├── PR_curve.png
    ├── labels_correlogram.png
    └── train_batch*.jpg   # Training preview images
```

You can use the trained model from:

```
project/Card-Model/weights/best.pt
```


