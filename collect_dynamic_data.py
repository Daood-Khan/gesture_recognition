import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Settings for gesture data collection
SEQUENCE_LENGTH = 30  # Number of frames per sequence
DATA_DIR = "dynamic_gesture_data"  # Directory to save gesture sequences

# Get user input for gesture name
gesture_name = input("Enter the name of the gesture you want to collect: ").strip()
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the directory exists

# Initialize webcam
cap = cv2.VideoCapture(0)

sequence = []  # Stores a single sequence of 30 frames
sequence_id = 0  # Sequence file numbering

print(f"Collecting data for gesture: {gesture_name}. Press 'r' to record and 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Process detected hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]  # Extract (x, y, z) coordinates

            # Draw landmarks on the frame for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Store landmarks until the sequence reaches the defined length
            if len(sequence) < SEQUENCE_LENGTH:
                sequence.append(landmarks)
            else:
                # Save the sequence as a .npy file
                file_path = os.path.join(DATA_DIR, f"{gesture_name}_{sequence_id}.npy")
                np.save(file_path, np.array(sequence))
                print(f"Saved sequence {sequence_id} for gesture {gesture_name}")

                # Reset sequence for next recording
                sequence_id += 1
                sequence = []

    # Display real-time recording info on the screen
    cv2.putText(frame, f"Gesture: {gesture_name} | Sequence ID: {sequence_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the video feed with hand tracking
    cv2.imshow('Dynamic Gesture Data Collector', frame)

    # Start recording with 'r', quit with 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("Recording sequence...")

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
hands.close()
