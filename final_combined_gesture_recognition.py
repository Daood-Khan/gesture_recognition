import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load trained gesture recognition models
static_model = load_model('model_static_gesture.keras')
dynamic_model = load_model('model_dynamic_gesture.keras')

# Load gesture label classes
static_classes = np.load('classes_static_label.npy')
dynamic_classes = np.load('classes_dynamic_label.npy')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize a buffer to store the last 40 frames of hand landmarks
sequence_buffer = deque(maxlen=40)

# Gesture recognition variables
gesture_mode = True  # True = static gesture mode, False = dynamic gesture mode
static_label = " "
dynamic_label = " "
stability_counter = 0
label_hold_counter = 0
label_hold_duration = 10
static_cooldown = 0  # Cooldown period for static gesture detection

# Motion thresholds
static_threshold = 0.0004
variance_check_frames = 10  # Number of frames to check for motion variance
high_motion_threshold = 0.002
stability_increment_threshold = 0.0002


def run_static_detection():
    """
    Perform static gesture classification using the last 40 frames of hand landmarks.
    Returns the predicted static gesture label.
    """
    global static_label

    # Flatten the latest 40-frame sequence for model input
    static_input = np.array(list(sequence_buffer)[-40:]).flatten()

    # Ensure the input shape matches the model's expected input size
    if static_input.shape[0] < 6300:
        static_input = np.pad(static_input, (0, 6300 - static_input.shape[0]), mode='constant')
    elif static_input.shape[0] > 6300:
        static_input = static_input[:6300]

    # Predict the static gesture class
    static_prediction = static_model.predict(np.expand_dims(static_input, axis=0))
    static_label = static_classes[np.argmax(static_prediction)]

    return static_label


def run_dynamic_detection():
    """
    Perform dynamic gesture classification using the last 30 frames.
    Returns the predicted dynamic gesture label with confidence.
    """
    global dynamic_label

    # Ensure there are at least 30 frames in the buffer before making a prediction
    if len(sequence_buffer) >= 30:
        input_sequence = np.array(list(sequence_buffer)[-30:])  # Get the latest 30 frames
        flattened_sequence = np.array([frame.flatten() for frame in input_sequence])  # Flatten each frame
        input_data = np.expand_dims(flattened_sequence, axis=0)  # Add batch dimension

        # Predict the dynamic gesture class
        prediction = dynamic_model.predict(input_data)
        confidence = np.max(prediction) * 100
        predicted_label = dynamic_classes[np.argmax(prediction)]
        dynamic_label = f"{predicted_label} ({confidence:.1f}%)"

        return dynamic_label
    else:
        return "No Gesture"


print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Process detected hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmark coordinates (x, y, z)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            sequence_buffer.append(landmarks)

            # Process gesture recognition when buffer is full
            if len(sequence_buffer) == 40:
                # Compute variance over the last few frames
                recent_frames = np.array(list(sequence_buffer)[-variance_check_frames:])
                variance = np.var(recent_frames, axis=0)
                mean_variance = np.mean(variance)

                print(f"Variance: {mean_variance}, Stability Counter: {stability_counter}")

                # Detect static gestures if motion is minimal and cooldown is over
                if mean_variance < stability_increment_threshold and static_cooldown == 0:
                    stability_counter += 1
                    if stability_counter >= 5:
                        static_label = run_static_detection()
                        label_hold_counter = label_hold_duration
                        static_cooldown = 15  # Apply cooldown to avoid frequent static detections

                # Detect dynamic gestures if motion is above a threshold
                elif mean_variance > high_motion_threshold:
                    stability_counter = 0  # Reset stability counter
                    dynamic_label = run_dynamic_detection()
                    label_hold_counter = label_hold_duration

                # Display recognized gesture on screen
                if label_hold_counter > 0:
                    label_hold_counter -= 1
                    if stability_counter >= 5:
                        cv2.putText(frame, f"Static Gesture: {static_label}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, f"Dynamic Gesture: {dynamic_label}", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Reduce static gesture cooldown counter
                if static_cooldown > 0:
                    static_cooldown -= 1

    # Display the video feed with gesture classification results
    cv2.imshow('Combined Gesture Recognition', frame)

    # Quit program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
hands.close()
