import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)

# Define Eye Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Drowsiness Parameters
EAR_THRESHOLD = 0.30  # Adjusted for broader detection
FRAME_THRESHOLD = 5   # Lowered for faster testing

frame_counter = 0
drowsy_alert = False

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

def play_beep():
    """Play a beep sound in a separate thread."""
    global drowsy_alert
    if not drowsy_alert:
        drowsy_alert = True
        try:
            # Generate a beep sound (1000 Hz, 1000 ms)
            winsound.Beep(1000, 1000)
        except Exception as e:
            print(f"Error playing beep: {e}")
        time.sleep(2)  # Cooldown to prevent rapid beeps
        drowsy_alert = False

def eye_aspect_ratio(eye_points, landmarks):
    """Calculate the Eye Aspect Ratio (EAR)."""
    A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    ear = (A + B) / (2.0 * C)
    return ear

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture video!")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    result = face_mesh.process(rgb_frame)

    landmarks_list = []
    if result.multi_face_landmarks:
        print("Face detected")
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks_list.append((x, y))

        if landmarks_list:
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks_list)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks_list)
            ear = (left_ear + right_ear) / 2.0

            # Debug EAR
            print(f"EAR: {ear:.2f}, Frame Counter: {frame_counter}")

            # Draw eye landmarks
            for point in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, landmarks_list[point], 2, (0, 255, 0), -1)

            # Drowsiness Detection
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= FRAME_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    threading.Thread(target=play_beep).start()
            else:
                frame_counter = 0

            # Display EAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        print("No face detected")

    # Show the frame
    cv2.imshow("Drowsiness Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()