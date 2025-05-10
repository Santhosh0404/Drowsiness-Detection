Drowsiness Detection System

A real-time drowsiness detection system using Python, OpenCV, and MediaPipe. It detects when a person’s eyes are closed for a prolonged period and plays a beep sound to alert them.

Installation





Clone the repository:

git clone https://github.com/Santhosh0404/Drowsiness-Detection.git
cd Drowsiness-Detection



Create a virtual environment:

python -m venv venv
.\venv\Scripts\activate  # On Windows



Install dependencies:

pip install -r requirements.txt

Usage





Ensure you have a webcam connected.



Run the script:

python drowsiness_detection.py



Close your eyes for a few seconds to trigger the alert. A beep sound will play, and "DROWSINESS DETECTED!" will appear on the video feed.



Press q to exit.

Dependencies





opencv-python



mediapipe



numpy



winsound (Windows only, part of Python standard library)
