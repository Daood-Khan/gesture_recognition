import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Directories for static and dynamic gesture datasets
STATIC_DATA_DIR = "static_gesture_data"
DYNAMIC_DATA_DIR = "dynamic_gesture_data"

# Output file mappings for static gestures
OUTPUT_STATIC = {
    "X_train": "static_X_train.npy",
    "X_test": "static_X_test.npy",
    "y_train": "static_y_train.npy",
    "y_test": "static_y_test.npy",
    "classes": "classes_static_label.npy"
}

# Output file mappings for dynamic gestures
OUTPUT_DYNAMIC = {
    "X_train": "dynamic_X_train.npy",
    "X_test": "dynamic_X_test.npy",
    "y_train": "dynamic_y_train.npy",
    "y_test": "dynamic_y_test.npy",
    "classes": "classes_dynamic_label.npy"
}


def process_static_data():
    """Preprocess and save static gesture data."""
    sequences = []  # Store flattened gesture frames
    labels = []  # Store gesture labels

    # Load and process static gesture data files
    for file in os.listdir(STATIC_DATA_DIR):
        if file.endswith(".npy"):
            file_path = os.path.join(STATIC_DATA_DIR, file)
            sequence = np.load(file_path)  # Load the stored gesture sequence
            sequence = sequence.flatten()  # Flatten all frames into a single feature vector
            sequences.append(sequence)
            labels.append(file.split("_")[0])  # Extract label from filename

    # Convert lists to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)

    # Encode string labels into numerical format
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save label classes for future reference
    np.save(OUTPUT_STATIC["classes"], le.classes_)

    # Split into training and testing sets if enough samples exist
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded
        print("Warning: Not enough samples for train-test split; using all data for training.")

    # Save preprocessed static gesture data
    np.save(OUTPUT_STATIC["X_train"], X_train)
    np.save(OUTPUT_STATIC["X_test"], X_test)
    np.save(OUTPUT_STATIC["y_train"], y_train)
    np.save(OUTPUT_STATIC["y_test"], y_test)

    print("Static gesture data preprocessing complete.")


def process_dynamic_data():
    """Preprocess and save dynamic gesture data."""
    sequences = []  # Store gesture sequences (multiple frames)
    labels = []  # Store gesture labels

    # Load and process dynamic gesture data files
    for file in os.listdir(DYNAMIC_DATA_DIR):
        if file.endswith(".npy"):
            file_path = os.path.join(DYNAMIC_DATA_DIR, file)
            sequence = np.load(file_path)  # Load stored sequence
            sequence = sequence.reshape(sequence.shape[0], -1)  # Flatten each frame
            sequences.append(sequence)
            labels.append(file.split("_")[0])  # Extract label from filename

    # Convert lists to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)

    # Encode string labels into numerical format
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save label classes for reference
    np.save(OUTPUT_DYNAMIC["classes"], le.classes_)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Save preprocessed dynamic gesture data
    np.save(OUTPUT_DYNAMIC["X_train"], X_train)
    np.save(OUTPUT_DYNAMIC["X_test"], X_test)
    np.save(OUTPUT_DYNAMIC["y_train"], y_train)
    np.save(OUTPUT_DYNAMIC["y_test"], y_test)

    print("Dynamic gesture data preprocessing complete.")


# Run preprocessing for both static and dynamic gesture datasets
if __name__ == "__main__":
    process_static_data()
    process_dynamic_data()
