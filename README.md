# AirDesk AI 🖐️

**AirDesk AI** is a touchless computer-control system that uses real-time hand gestures to interact with a computer through the camera.

Instead of using a traditional mouse for basic actions, AirDesk uses computer vision and hand landmarks to detect gestures and translate them into actions such as cursor movement, clicking, scrolling, enabling controls, pausing controls, and stopping the system.

## ✨ Features

* Real-time hand tracking
* Gesture recognition using MediaPipe
* Touchless cursor control
* Left-click using pinch gesture
* Scrolling using Peace gesture
* Control enable/disable system
* Safety pause gesture
* Gesture smoothing for more stable detection
* FPS monitoring
* Live vision dashboard
* Two-hand detection support
* Apple Silicon / macOS compatible setup

## ✋ Gesture Controls

| Gesture      | Action          |
| ------------ | --------------- |
| ☝️ Point     | Cursor movement |
| 🤏 Pinch     | Left click      |
| ✌️ Peace     | Scroll          |
| 👍 Thumbs Up | Enable controls |
| ✋ Open Palm  | Safety pause    |
| ✊ Fist       | Stop controls   |

## 🛠️ Technology Stack

* **Python**
* **OpenCV**
* **MediaPipe**
* **PyAutoGUI**
* **NumPy**
* **Computer Vision**
* **Hand Landmark Detection**

## 📁 Project Structure

```text
AirDesk/
├── main.py
├── air_control.py
├── camera_test.py
├── dashboard.py
├── gesture_engine.py
├── hand_tracking.py
├── models/
│   └── hand_landmarker.task
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Setup

Clone the repository:

```bash
git clone https://github.com/pavitra-G16/AirDesk.git
cd AirDesk
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run AirDesk

```bash
python main.py
```

Make sure the application has permission to access the camera.

Press **Q** to safely stop AirDesk.

## 🧠 How It Works

```text
Camera
   ↓
OpenCV
   ↓
MediaPipe Hand Detection
   ↓
Hand Landmarks
   ↓
Gesture Recognition
   ↓
Gesture Smoothing
   ↓
Air Control Engine
   ↓
Computer Action
```

AirDesk captures frames from the camera, converts them into RGB images, and processes them using MediaPipe's hand-landmark detection.

The detected landmarks are analyzed to identify gestures. A smoothing mechanism helps reduce accidental gesture changes caused by individual unstable frames.

The recognized gesture is then converted into a computer action.

## 🔒 Safety

AirDesk includes safety-oriented controls to prevent accidental interaction.

* **Open Palm** → pauses controls
* **Fist** → stops controls
* **Q key** → safely exits the application

The system is designed so that computer control can be disabled when required.

## 🎥 Demo

A demonstration video can be added here:

**Demo Video:**
*Add your Google Drive / YouTube demo link here.*

## 📸 Screenshots

*Add screenshots of the AirDesk interface and hand-tracking system here.*

## 🎯 Project Goal

The goal of AirDesk AI is to explore accessible and touchless human-computer interaction using computer vision.

The project demonstrates how hand landmarks and gesture recognition can be combined with desktop automation to create a practical AI-powered interface.

## 👨‍💻 Developer

**Pavitra Gangwar**

AI & Data Science Student | Python | Computer Vision | AI Product Builder

## 📄 License

This project is intended for educational, experimental, and portfolio purposes.
grep -n "GESTURE\|gesture\|PINCH\|PEACE\|THUMB\|FIST\|PALM" main.py | head -40

