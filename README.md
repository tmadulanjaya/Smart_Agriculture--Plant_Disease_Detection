# 🌿 Plant Disease Detection using Computer Vision

## 📋 Project Overview

An automated plant disease detection system built using deep learning and computer vision techniques. The system classifies leaf images into **38 plant disease categories** across **14 plant species** with a test accuracy of **95.33%**.

## 📁 Repository Structure

📦 plant-disease-detection
 ┣ 📓 smart-agriculture.ipynb     ← Main Jupyter Notebook (run on Kaggle)
 ┣ 📄 class_names.json            ← List of 38 disease class names
 ┗ 📄 README.md                   ← This file
 
## 🗂️ Dataset

**PlantVillage Dataset** — 54,305 leaf images across 38 disease classes

| Property | Value |
|----------|-------|
| Total Images | 54,305 |
| Disease Classes | 38 |
| Plant Species | 14 |
| Training Set | 43,444 images (80%) |
| Validation Set | 5,430 images (10%) |
| Test Set | 5,430 images (10%) |

**Source:** [Kaggle - abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## 🧠 Model Architecture

A custom **4-block Convolutional Neural Network (CNN)** trained from scratch:

```
Input (128x128x3)
    ↓
Conv2D(32) + BatchNorm + MaxPool
    ↓
Conv2D(64) + BatchNorm + MaxPool
    ↓
Conv2D(128) + BatchNorm + MaxPool
    ↓
Conv2D(256) + BatchNorm + MaxPool
    ↓
Flatten → Dense(512) + Dropout(0.5)
    ↓
Dense(38) + Softmax → Disease Class
```

---

## 🖼️ Image Processing Techniques

Six computer vision techniques were applied:

| Technique | Purpose |
|-----------|---------|
| Grayscale Conversion | Reduce image complexity |
| Histogram Equalization | Enhance disease contrast |
| Gaussian Blur | Remove noise |
| Canny Edge Detection | Extract disease boundaries |
| HSV Colour Segmentation | Isolate diseased regions |
| Data Augmentation | Improve generalisation |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **95.33%** |
| Test Loss | 0.2194 |
| Best Validation Accuracy | 95.13% (Epoch 24) |
| Macro Average F1-Score | 0.94 |
| Weighted Average F1-Score | 0.95 |
| Training Epochs | 25 |
| Training Hardware | Dual Tesla T4 GPU (Kaggle) |

---

## 🚀 How to Run

### Option 1 — Run on Kaggle (Recommended)

1. Go to [Kaggle.com](https://kaggle.com) and sign in
2. Create a new notebook
3. Add the PlantVillage dataset: `abdallahalidev/plantvillage-dataset`
4. Upload `smart-agriculture.ipynb`
5. Set accelerator to **GPU T4 x2**
6. Run all cells

### Option 2 — Run Locally

```bash
# Install dependencies
pip install tensorflow opencv-python scikit-image scikit-learn matplotlib seaborn numpy

# Update DATA_DIR in notebook to your local dataset path
# Then run all cells in Jupyter Notebook
```

### Option 3 — Load Pretrained Model

```python
from tensorflow.keras.models import load_model
import json, cv2, numpy as np

# Download best_model.keras from OneDrive link above
model = load_model("best_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Predict on a new image
def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img / 255.0, axis=0)
    pred = model.predict(img, verbose=0)
    print(f"Disease    : {class_names[np.argmax(pred)]}")
    print(f"Confidence : {np.max(pred)*100:.2f}%")

predict("your_leaf_image.jpg")
```


## 🛠️ Technologies Used

- **Python** 3.10+
- **TensorFlow / Keras** 2.19.0
- **OpenCV** 4.13.0
- **scikit-learn** — class weights & evaluation
- **scikit-image** — image analysis
- **NumPy** 2.0.2
- **Matplotlib / Seaborn** — visualisation
- **Kaggle Notebooks** — GPU training environment
