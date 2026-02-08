# 🎭 Real-Time Face Recognition & Emotion Detection System

A real-time **Face Recognition and Emotion Detection** application built using **Python, OpenCV, and DeepFace**.  
The system captures live webcam video, detects human faces, and predicts the **dominant emotion** (happy, sad, angry, neutral, etc.) for each detected face instantly.

This project demonstrates practical implementation of **Computer Vision + Deep Learning** and can be used in smart attendance systems, surveillance, human-computer interaction, and behavioral analytics.

---

## 🚀 Features

- Real-time face detection using OpenCV Haar Cascades
- Live webcam video processing
- Emotion detection using DeepFace deep learning models
- Face cropping and preprocessing
- Dataset collection (captures up to 100 face samples)
- Displays dominant emotion on screen
- Bounding box around detected face
- Frame counter showing captured samples

---

## 🧠 Technologies Used

- **Python**
- **OpenCV (cv2)** – Face detection & video processing
- **DeepFace** – Deep learning based emotion recognition
- **NumPy** – Image data handling
- **Scikit-image (SSIM)** – Image similarity metrics
- **Haar Cascade Classifier** – Face detection model

---

## 📂 Project Workflow

1. Webcam starts and captures live video
2. Each frame is converted to grayscale
3. Haar Cascade detects face(s)
4. Face region is cropped and resized (200x200)
5. DeepFace analyzes the face
6. Dominant emotion is predicted
7. Emotion label appears below face in real-time

---

## 🖥️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/face-emotion-recognition.git
cd face-emotion-recognition
```

### 2️⃣ Create virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install opencv-python
pip install numpy
pip install scikit-image
pip install deepface
pip install tensorflow
```

---

## ▶️ Running the Project

Run the Python file:

```bash
python main.py
```

Your webcam will open and start detecting faces and emotions.

Press **Q** to quit the application.

---

## 📸 Output

The system will:

- Detect your face
- Draw a rectangle around it
- Display detected emotion (Happy / Sad / Angry / Neutral / Surprise / Fear / Disgust)

Example:

```
[Face Detected] → Emotion: Happy
```

---

## ⚙️ How It Works

### Face Detection
The model uses:

```
haarcascade_frontalface_default.xml
```

OpenCV scans each frame and finds facial patterns like eyes, nose, and mouth.

### Emotion Detection
We use **DeepFace**, which internally uses pre-trained CNN models trained on large facial expression datasets.

It analyzes facial regions and outputs the **dominant emotion**.

---

## 📊 Dataset Collection (Built-in Feature)

The program automatically collects face images:

- Captures every 10th frame
- Stores up to 100 face samples
- Can later be used for training a custom recognition model

---

## 🔮 Future Improvements

- Face recognition (identify specific person)
- Attendance system integration
- Save detected emotions to CSV
- GUI interface (Tkinter / PyQt)
- Multiple face tracking
- Deploy as a web app (Flask/Streamlit)

---

## 💡 Applications

- Smart attendance systems
- Surveillance & security
- Online exam monitoring
- Human behavior analysis
- Human-Computer Interaction

---

## 👩‍💻 Author

**Lovely Bisht**  

