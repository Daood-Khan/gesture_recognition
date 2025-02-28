import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Get user input for gesture label and number of frames to record
gesture = input("Enter gesture label: ")
num_frames = int(input("Enter the number of frames to record: "))

# Create a directory for saving gesture data if it doesn't exist
os.makedirs("static_gesture_data", exist_ok=True)

# Check for existing gesture files
existing_files = [
    f for f in os.listdir("static_gesture_data") 
    if f.startswith(gesture) and f.endswith(".npy")
]

if existing_files:
    # Find the highest-numbered file for the gesture and increment the filename
    highest_number = max(
        [
            int(f.split("_")[-1].split(".")[0]) 
            for f in existing_files 
            if "_" in f and f.split("_")[-1].split(".")[0].isdigit()
        ],
        default=0
    )
    new_file_name = f"{gesture}_{highest_number + 1}.npy"
else:
    new_file_name = f"{gesture}_1.npy"

# Initialize a list to store hand landmark frames
frames = []
count = 0  # Counter for recorded frames

print(f"Recording {num_frames} frames for gesture: {gesture}. Press 'q' to quit early.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Process detected hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract (x, y, z) coordinates for each landmark
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            frames.append(landmarks)  # Store landmarks for this frame
            count += 1
            print(f"Captured frame {count}")

    # Display the frame with landmarks
    cv2.imshow('Static Gesture Data Collection', frame)

    # Stop recording if 'q' is pressed or the required frames are captured
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_frames:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

# Save or append collected gesture data
save_path = f"static_gesture_data/{gesture}.npy"
if os.path.exists(save_path):
    # Append new data to the existing file
    existing_data = np.load(save_path)
    frames = np.concatenate((existing_data, frames), axis=0)
    np.save(save_path, frames)
    print(f"Appended {count} frames to existing gesture data: {save_path}")
else:
    # Save as a new file with an incremented name
    np.save(f"static_gesture_data/{new_file_name}", np.array(frames))
    print(f"Saved {count} frames for gesture '{gesture}' in static_gesture_data/{new_file_name}")
