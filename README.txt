GESTURE RECOGNITION PROJECT
====================================

This project implements a real-time **gesture recognition system** using computer vision 
and deep learning. It supports both **static** and **dynamic** gesture recognition using 
pre-trained models, allowing direct execution without additional setup. 

The goal of this project is to provide an alternative communication method through 
gesture-based interaction. This proof-of-concept demonstrates how hand gestures can be 
interpreted in real time using AI.

------------------------------------
FEATURES
------------------------------------
- Recognizes **static** and **dynamic** hand gestures.
- Uses **separate models** for static and dynamic gestures.
- **Pre-trained models included** (No need to train from scratch).
- Implements a **stability counter** to reduce false detections and improve accuracy.
- Outputs **real-time gesture classifications**.

Current Supported Gestures:
- **Static:** "Peace" sign
- **Dynamic:** "Wave" gesture

------------------------------------
SETUP INSTRUCTIONS
------------------------------------
1. INSTALL DEPENDENCIES  
   Ensure you have Python installed, then install the required libraries:
   

pip install -r requirements.txt


2. RUN GESTURE RECOGNITION  
Start the real-time gesture recognition system:

python final_combined_gesture_recognition.py


This will activate the camera and begin detecting static and dynamic gestures.

------------------------------------
PROJECT STRUCTURE
------------------------------------
MAIN FILES:
- **final_combined_gesture_recognition.py**  -> Final working version (use this).
- **requirements.txt**                         -> List of dependencies for easy installation.
- **README.txt**                               -> Project documentation.

DATA COLLECTION:
- **collect_static_data.py**                   -> Collects static gesture data.
- **collect_dynamic_data.py**                   -> Collects dynamic gesture data.

PREPROCESSING:
- **preprocessing.py**                        -> Converts collected data into a suitable format for training.

MODEL TRAINING:
- **train_static_model_final.py**              -> Trains the static gesture model.
- **train_dynamic_model_final.py**             -> Trains the dynamic gesture model.

PRE-TRAINED MODELS & CLASS LABELS:
- **model_static_gesture.keras**               -> A Trained model for 1 static gesture (Peace).
- **model_dynamic_gesture.keras**              -> A Trained model for 1 dynamic gesture (Wave).
- **classes_static_label.npy**                 -> Label encoding for static gestures.
- **classes_dynamic_label.npy**                -> Label encoding for dynamic gestures.

------------------------------------
TRAINING YOUR OWN MODELS
------------------------------------
If you want to train your own models instead of using the pre-trained ones, follow these steps:

1. **Collect Gesture Data:**
- Run `collect_static_data.py` or 'collect_dynamic_data.py` to collect the respective gesture data
- Name the gesture and select how many times/frames you want to record 
- A folder will be created for the respective type of gesture if it doesn't already exist, training new gestures adds to said folder

2. **Preprocess Data:**
- Run `preprocessing.py` to prepare the dataset for training.

3. **Train the Models:**
- Run `train_static_model_final.py` to train a new static gesture model.
- Run `train_dynamic_model_final.py` to train a new dynamic gesture model.
- After training, rename `model_static_gesture.keras` and `model_dynamic_gesture.keras` 
  if the a model with this name exists it will replace it.

------------------------------------
NOTES
------------------------------------
- If a script fails due to missing dependencies, ensure `requirements.txt` is up to date.
- If modifying models, retrain them and update the `.keras` files accordingly.
- Older versions & test scripts have been removed to keep the repository clean.

------------------------------------
FUTURE IMPROVEMENTS
------------------------------------
- Add support for **more gestures**.
- Optimize real-time detection for **better accuracy & performance**.
- Implement gesture-based **control for applications**.
- Improve **dynamic gesture classification** to distinguish similar movements.

------------------------------------
AUTHOR
------------------------------------
Developed by **Daood Khan**

This project is a proof-of-concept demonstrating how computer vision and AI 
can be leveraged for gesture-based interaction. 

Enjoy! :)
