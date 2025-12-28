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
<img width="252" height="687" alt="image" src="https://github.com/user-attachments/assets/847b5fb2-d628-4835-9595-b9a92875de5a" />



---

## ⚙️ Installation

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux
venv\Scripts\activate      # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt


If requirements.txt is not available:

pip install torch torchvision numpy opencv-python pillow matplotlib tqdm tensorboard streamlit

📊 Dataset Preparation (FER2013)

If the dataset is provided as a CSV file:

from preprocessing import FERDatasetLoader
FERDatasetLoader.csv_to_images("fer2013.csv", "fer2013")


This will generate:

fer2013/
├── train/
└── test/

🧠 Model Architecture

Supported Models:

ResNet18 (default)

MobileNetV2

Final classifier:

nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(in_features, 7)
)

🏋️ Training the Model
python train.py

Training Features

AMP (Mixed Precision)

Adam Optimizer

ReduceLROnPlateau Scheduler

Early Stopping

Checkpoint Saving

TensorBoard Logging

View training logs:

tensorboard --logdir results/logs

📈 Evaluation
python evaluate.py


Generates:

Classification Report

Confusion Matrix

Accuracy & Loss Metrics

Saved plots in results/plots/

🔍 Prediction
Single Image
python predict.py --image path/to/image.png


Example Output:

Prediction: Happy (94.12%)

🎥 Real-Time Webcam Detection
python live_detection.py


Controls:

q → Quit

s → Save frame

r → Record video

Features:

Real-time emotion detection

FPS display

Emotion smoothing

🖥️ Graphical User Interface (GUI)

A fully interactive Streamlit-based web GUI with modern UI/UX design.

GUI Capabilities

Upload image & predict emotion

Live webcam emotion detection

Model inspection

Dataset exploration

Training & evaluation visualization

Dark / Light mode

Animated and responsive UI

Run GUI
streamlit run app.py

🧩 GUI Architecture
app.py
pages/
├── image_prediction.py
├── webcam_live.py
├── model_info.py
├── dataset_page.py
└── evaluation_page.py
assets/
└── fer_example.jpg

🚀 Model Improvements & Enhancements
Baseline Performance

FER2013 only: ~65% accuracy (expected for dataset quality)

Identified Limitations

Noisy labels

Low resolution images

Class imbalance

Enhancement Strategy

Multi-dataset training:

FER2013

FER+

AffectNet

Stronger models:

ResNet34

CBAM Attention Module

Final Outcomes

Higher accuracy

Better feature discrimination

Improved real-time stability

👥 Team Members & Contributions

Hazem Hatem

Data preprocessing & augmentation

Model information module

Ali Waheed Abdullah

Model architecture (model.py)

Main GUI (app.py)

Abdullah El-Shahaly

Training pipeline (train.py)

Dataset GUI module

Fares Ashraf

Evaluation pipeline (evaluate.py)

Evaluation GUI module

Youssef Ibrahim Abd Elwahab

Image prediction logic

Image prediction GUI

Eslam Ayman Kamal

Real-time detection pipeline

Webcam GUI module

## 📜 License

This project is developed for educational and research purposes.

