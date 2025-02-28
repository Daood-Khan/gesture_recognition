# Gesture Recognition Project

This project implements a gesture recognition system using computer vision and machine learning. 
It supports both **static** and **dynamic** gesture recognition, utilizing deep learning models 
trained on collected gesture data.

Note on Redundant Files:
(I HAVE LEFT IN REDUNDANT FILES TO SHOW MY ATTEMPTS THAT FAILED TO SHOW MY DEVELOPMENT PROGRESS.)
Some files may be older versions or redundant attempts that didn’t work as expected. These have been 
retained intentionally to document my iteration process and problem-solving steps. The folder older 
versions that don't work/ contains many of these, but some others may still be in the main directory.

## Features
- Recognizes **static** and **dynamic** hand gestures.
- Uses **separate models** for static and dynamic gestures.
- Data preprocessing and training scripts included.
- Implements a **stability counter** to refine gesture detection.
- Outputs real-time gesture classifications.

---
## Setup Instructions
### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```
pip install -r requirements.txt
```

### 2. Running Gesture Recognition
To run the final integrated system:
```
python main.py
```

To test static gesture detection separately:
```
python static_detection_testing_for_final_main.py
```

To test dynamic gesture recognition:
```
python dynamic_gesture_recognition.py
```

---
## Project Structure
### Main Files
- **main.py** - an attempt on the full gesture recognition system.
- **final.py** - Another main script attempt (ignore).
- **combined_gesture_recognition- FINAL WORKING PROJECT.py** -  the final integrated version.

### Gesture Recognition
- **static_detection_testing_for_final_main.py** - Tests static gesture detection separately.
- **dynamic_gesture_recognition.py** - Runs dynamic gesture detection.
- **gesture_classes.npy** - Contains gesture class labels.

### Data Collection
- **data_collector.py** - Handles gesture data collection.
- **dynamic_data_collector.py** - Collects data specifically for dynamic gestures.
- **static_gesture_data/** - Stores static gesture data.
- **dynamic_gesture_data/** - Stores dynamic gesture data.
- **older versions that dont work/** -  contains old/incomplete versions.

### Preprocessing
- **preprocessing.py** - Preprocesses both static and dynamic gesture data.
- **preprocess_static.py** - Preprocesses static gestures.
- **preprocess_dynamic.py** - Preprocesses dynamic gestures.
- **preprocess_dynamic_data2.py** - Another dynamic gesture preprocessing script.

### Model Training
- **train_static_model_final.py** - Trains the static gesture recognition model.
- **train_dynamic_model_final.py** - Trains the dynamic gesture recognition model.
- **train_model.py** - General training script (an older version).
- **train_dynamic_model.py** - Another dynamic model training script.

### Models and Label Files
- **static_gesture_model.keras** - Saved model for static gestures.
- **dynamic_gesture_model.keras** - Saved model for dynamic gestures.
- **gesture_model.keras** - Another saved model (old version).
- **static_label_classes.npy** - Stores label encoding for static gestures.
- **dynamic_label_classes.npy** - Stores label encoding for dynamic gestures.
- **label_encoder_classes.npy** - General label encoding file.

### Testing and Debugging
- **tester.py** -  a testing script for mediapipe.
- **test1.py, test2.py, test3.py** - more test scripts.

### Data Files
- **X.npy, y.npy** - General dataset files.
- **X_train.npy, X_test.npy, y_train.npy, y_test.npy** - Training/testing datasets for gesture recognition.
- **static_X_train.npy, static_X_test.npy, static_y_train.npy, static_y_test.npy** - Static gesture dataset.
- **dynamic_X_train.npy, dynamic_X_test.npy, dynamic_y_train.npy, dynamic_y_test.npy** - Dynamic gesture dataset.

---
## Notes
- If any script fails due to missing dependencies, ensure `requirements.txt` is up to date.
- If modifying the models, update the corresponding `.keras` files after retraining.
- Old versions of scripts are stored in the `older versions that dont work/` folder for reference.

---
## Future Improvements
- Optimize real-time gesture recognition for **better accuracy and stability**.
- Implement **gesture-based control** for real-world applications.
- Improve **dynamic gesture classification** to differentiate similar gestures better.

---
## Author
Daood Khan (as seen in `Dev Log.pdf`).

enjoy :)
