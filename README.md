# Facial-expression-detection-updated

# 🎭 Real-Time Facial Emotion Recognition (FER2013)

## 📌 Project Overview

This project detects human facial emotions in real-time using a webcam.
A Convolutional Neural Network (CNN) is trained on the **FER2013 dataset** and integrated with **OpenCV** for live face detection and emotion prediction.

The system identifies 7 emotions:

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😄
* Sad 😢
* Surprise 😮
* Neutral 😐

---

## 🧠 Project Workflow

1. Train CNN model using FER2013 dataset (Google Colab GPU)
2. Download trained model to local machine
3. Detect face using OpenCV Haar Cascade
4. Predict emotion using trained CNN
5. Display emotion in real-time via webcam

---

## ⚙️ Technologies Used

* Python 3.10
* TensorFlow / Keras
* OpenCV
* NumPy
* Google Colab (for GPU training)

---

## 📂 Project Structure

```
FER_Project
│
├── detect_emotion.py
├── emotion_model.hdf5
├── haarcascade_frontalface_default.xml
└── venv/
```

---

## 🚀 Step 1 — Train Model on Google Colab (GPU)

### 1️⃣ Enable GPU

Top menu:

Runtime → Change runtime type → GPU → Save
OR
Runtime → Change runtime type → Select **T4 GPU**

Check GPU:

import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))


### 2️⃣ Upload FER2013 Dataset (Image Version)

Download FER2013 dataset from kaggle
https://www.kaggle.com/datasets/msambare/fer2013?resource=download

Upload the ZIP to Colab

Left sidebar → Files → Upload → upload the entire fer2013 zip

After upload, run this cell:

import zipfile

zip_path = "archive.zip"   # rename if your zip name is different
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()

print("Dataset extracted")


After extraction you should see:

train/
test/

Dataset structure used:

```
train/
test/
```

### 3️⃣ Train CNN Model

Model trained for **12 epochs** using data augmentation.

Expected accuracy:

```
50–60% (FER2013 is a difficult dataset)
```

### 4️⃣ Save and Download Model

```
emotion_model.hdf5
```

---

All steps are done in this notebook , you can make use of this.
https://colab.research.google.com/drive/1vtAyl9wDDHQ54Sp-cnB1ueyCsiQ563Xm?usp=sharing

## 💻 Step 2 — Local Setup (Windows)

### Install Python Version

TensorFlow requires Python **3.10**

Download:
https://www.python.org/downloads/release/python-3109/

---

## 🐍 Step 3 — Create Virtual Environment

Open terminal inside project folder:

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

---

## 📦 Step 4 — Install Dependencies

```bash
pip install tensorflow opencv-python numpy
```

---

## 📥 Step 5 — Download Haar Cascade

Download:
https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

Place in project folder.

---

## ▶️ Step 6 — Run the Project

Activate environment:

```bash
venv\Scripts\activate
```

Run:

```bash
python detect_emotion.py
```

Press **Q** to exit webcam.

---

### Output
![WhatsApp Image 2026-03-02 at 6 31 31 AM](https://github.com/user-attachments/assets/a05d9b62-043f-414a-b2c7-56ba1f8bc6ca)


## ⚡ Performance Optimizations Implemented

### Faster Camera Opening

Used DirectShow backend:

```python
cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

### Stable Emotion Prediction

Emotion updates every **0.7 seconds** to avoid flickering:

```python
if time.time() - last_prediction_time > 0.7:
```

---

## 🎭 How to Demonstrate Emotions

| Emotion  | Expression Tips             |
| -------- | --------------------------- |
| Happy    | Big smile with teeth        |
| Sad      | Droopy eyes, pout lips      |
| Angry    | Eyebrows down, tight lips   |
| Surprise | Open mouth, raised eyebrows |
| Fear     | Wide eyes, tense mouth      |
| Disgust  | Wrinkled nose               |
| Neutral  | Relaxed face                |

---

## 🏁 Final Result

✔ Real-time face detection
✔ Real-time emotion prediction
✔ Stable and smooth output
✔ Fully working ML project

---

## 📚 Future Improvements

* Improve accuracy to 70%+
* Add emotion confidence percentage
* Deploy as web app
* Use deep face detection (DNN/MTCNN)

---

## 👨‍💻 Author

Vaibhav Pingale

---
