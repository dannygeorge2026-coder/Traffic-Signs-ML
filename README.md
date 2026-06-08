# 🚦 Traffic Sign Classifier — GTSRB

A convolutional neural network (CNN) trained on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/) to classify 43 categories of road signs.

---

## Overview

This project builds and trains a CNN in PyTorch to recognise German traffic signs from images. The model is trained on the GTSRB dataset, which contains over 50,000 images across 43 classes, and achieves solid accuracy on the held-out test set.

---

## Features

- Custom CNN architecture built with `torch.nn`
- Trained on the official GTSRB train/test splits via `torchvision.datasets.GTSRB`
- 43-class classification (speed limits, warnings, prohibitions, mandatory signs, etc.)
- Inference support for images loaded from a URL
- Model weights saved to `.pth` for reuse

---

## Model Architecture

```
Input: 32×32 RGB image

Conv2d(3 → 32, 3×3) → ReLU → MaxPool(2×2)
Conv2d(32 → 64, 3×3) → ReLU → MaxPool(2×2)
Flatten
Linear(64×8×8 → 128) → ReLU
Linear(128 → 43)
```

---

## Requirements

```
torch
torchvision
matplotlib
Pillow
requests
tqdm
```

Install with:

```bash
pip install torch torchvision matplotlib Pillow requests tqdm
```

---

## Usage

### Train the model

```bash
python signs.py
```

This will:
1. Download the GTSRB dataset into `./data/`
2. Train the CNN for 15 epochs (batch size 64, Adam optimiser, lr=0.001)
3. Print train/test loss and accuracy per epoch
4. Save weights to `gtsrb_model.pth`
5. Run a sample prediction from a URL

### Run inference on a custom image

Modify the `image_url` variable at the bottom of `signs.py` to point to any publicly accessible traffic sign image:

```python
image_url = "https://your-image-url-here.jpg"
preprocessed_image_tensor = load_and_preprocess_image_from_url(image_url, transform)
predicted_class_name = predict_image_class(model_1, preprocessed_image_tensor, class_names)
print(f"The predicted traffic sign is: {predicted_class_name}")
```

---

## Classes

The model recognises 43 traffic sign types, including:

| ID | Class |
|----|-------|
| 0 | Speed limit (20 km/h) |
| 14 | Stop |
| 17 | No entry |
| 25 | Road work |
| 33 | Turn right ahead |
| 38 | Keep right |
| ... | *(see `signs.py` for full list)* |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Input size | 32×32 RGB |

---

## Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**
- ~39,000 training images, ~12,600 test images
- 43 classes
- Downloaded automatically via `torchvision.datasets.GTSRB`

---

## License

This project is for educational purposes. The GTSRB dataset is provided by the Institut für Neuroinformatik, Ruhr-Universität Bochum.
