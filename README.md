🖊 Air Writing with Hand Gestures

This project allows you to draw in the air using your fingers.
It uses MediaPipe for hand-tracking and OpenCV for visualization.

🚀 Features

Detects your hand using MediaPipe Hands
Tracks index finger tip for drawing
Supports multiple drawing colors
Can clear the screen and restart

🛠 Requirements
Python 3.10 (⚠ Mediapipe doesn’t support Python 3.13+)
pip package manager

📥 Installation
1. Install Python 3.10
Download and install from:
👉 Python 3.10.11 (Windows 64-bit)
✅ Tick “Add Python to PATH” during installation.

Check version:
py -3.10 --version

2. Create Virtual Environment (Optional but Recommended)
py -3.10 -m venv venv
venv\Scripts\activate   # On Windows

3. Install Dependencies
py -3.10 -m pip install mediapipe opencv-python numpy
▶ Usage
Run the project:
py -3.10 air-writing.py

🎨 Controls
Use Index Finger → Draw on screen
Use Thumb + Index Finger together → Clear screen
Press Q → Quit

📸 Demo
checkout linedin: https://www.linkedin.com/in/poondlamanasa
