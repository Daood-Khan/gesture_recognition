import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load preprocessed dynamic gesture data
X_train = np.load("dynamic_X_train.npy")
X_test = np.load("dynamic_X_test.npy")
y_train = np.load("dynamic_y_train.npy")
y_test = np.load("dynamic_y_test.npy")

# Convert class labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the LSTM model for dynamic gesture recognition
model = Sequential([
    # First LSTM layer with return_sequences=True to pass output to the next LSTM layer
    LSTM(128, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.4),  # Dropout to prevent overfitting

    # Second LSTM layer without returning sequences (final LSTM layer)
    LSTM(256, return_sequences=False, activation='relu'),
    Dropout(0.4),  # Additional dropout layer

    # Fully connected layer with ReLU activation
    Dense(128, activation='relu'),

    # Output layer with softmax activation for multi-class classification
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with training data and validate with test data
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model for future use
model.save("model_dynamic_gesture.keras")
print("Dynamic gesture model trained and saved as 'model_dynamic_gesture.keras'")
