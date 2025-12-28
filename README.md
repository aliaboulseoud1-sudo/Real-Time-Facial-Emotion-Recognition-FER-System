# 🧠 Real-Time Facial Emotion Recognition (FER) System

Deep Learning Project using **PyTorch**  
Transfer Learning (ResNet18 / MobileNetV2) + **Streamlit GUI**

📅 Date: 11/12/2025  
👥 Team: **THE BRO**

---

## 📌 Overview
This project implements a complete **Real-Time Facial Emotion Recognition (FER)** system trained on the **FER2013** dataset.  
It provides an end-to-end deep learning pipeline starting from data preprocessing to real-time webcam emotion detection with a modern web-based GUI.

The system is modular, configurable, and production-ready.

---

## ✨ Key Features
- Data preprocessing & augmentation
- Custom dataset loader
- Transfer Learning using **ResNet18** and **MobileNetV2**
- Mixed Precision Training (AMP)
- Learning rate scheduling
- Early stopping
- Model evaluation (confusion matrix & metrics)
- Image & batch predictions
- Real-time webcam emotion detection
- Centralized configuration via `config.py`
- Fully interactive **Streamlit GUI**

---

## 📂 Project Structure
```bash
deep_learning_project/
│── config.py
│── preprocessing.py
│── model.py
│── train.py
│── evaluate.py
│── predict.py
│── live_detection.py
│── checkpoints/
│── results/
│ ├── logs/
│ ├── plots/
│ ├── metrics/
│ ├── predictions/
│ ├── screenshots/
│ └── recordings/
│── fer2013/
│ ├── fer2013.csv
│ ├── train/
│ └── test/
└── gui/
├── app.py
├── pages/
│ ├── image_prediction.py
│ ├── webcam_live.py
│ ├── model_info.py
│ ├── dataset_page.py
│ └── evaluation_page.py
```


---

## ⚙️ Installation

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux
venv\Scripts\activate      # Windows
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
If requirements.txt is not available:
```bash
pip install torch torchvision numpy opencv-python pillow matplotlib tqdm tensorboard streamlit
```
---
## 📊 Dataset Preparation (FER2013)

If the dataset is provided as a CSV file:
```python
from preprocessing import FERDatasetLoader
FERDatasetLoader.csv_to_images("fer2013.csv", "fer2013")
```

This will generate:
```python
fer2013/
├── train/
└── test/
```
---
## 🧠 Model Architecture

### Supported Models
- **ResNet18** (default)
- **MobileNetV2**

### Final Classifier
```python
nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(in_features, 7)
)
```
