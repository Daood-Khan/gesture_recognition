from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# Load preprocessed static gesture data
X_train = np.load("static_X_train.npy")
X_test = np.load("static_X_test.npy")
y_train = np.load("static_y_train.npy")
y_test = np.load("static_y_test.npy")

# Verify labels and determine the number of unique classes
unique_labels = np.unique(y_train)
print(f"Unique labels in y_train: {unique_labels}")  # Debugging step to check class labels
num_classes = int(np.max(y_train)) + 1  # Determine the number of classes from the max label value

# Convert class labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Define the feedforward neural network model for static gesture classification
model = Sequential([
    # First fully connected layer with ReLU activation
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Dropout to prevent overfitting

    # Second fully connected layer
    Dense(64, activation='relu'),
    Dropout(0.3),  # Additional dropout for regularization

    # Output layer with softmax activation for multi-class classification
    Dense(num_classes, activation='softmax')  # Number of classes dynamically determined
])

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with training data and validate using test data
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model for future use
model.save("model_static_gesture.keras")
print("Static gesture model trained and saved as 'model_static_gesture.keras'")
