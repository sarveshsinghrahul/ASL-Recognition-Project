# receiver_landmark.py
import cv2
import zmq
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque # <-- 1. ADD THIS

# --- 1. LOAD YOUR *NEW* LANDMARK MODEL ---
try:
    asl_model = load_model('asl_landmark_model.keras')
    print("LANDMARK model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. DEFINE CLASS NAMES (Must match LabelEncoder) ---
# You can get this from your notebook's `label_encoder.classes_`
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]

# --- 3. SET UP ZMQ RECEIVER ---
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "") 
print("Receiver (Landmark) is running...")

# --- !!! NEW CODE TO FIX JITTER !!! ---
# 1. Create a "deque" (a fast list) to hold the last 10 predictions
predictions_deque = deque(maxlen=10) 
# --- END NEW CODE ---

while True:
    try:
        data_bytes = socket.recv()
        landmark_vector = np.frombuffer(data_bytes, dtype=np.float64) 
        model_input = landmark_vector.reshape(1, -1) 
        
        # --- 5. MAKE A PREDICTION ---
        prediction = asl_model.predict(model_input, verbose=0)
        pred_index = np.argmax(prediction[0])
        confidence = prediction[0][pred_index]
        
        current_label = ""
        
        # --- !!! NEW CODE TO FIX JITTER !!! ---
        # 2. Add the prediction index to our list
        predictions_deque.append(pred_index)
        
        # 3. Find the most common prediction in the list
        # We check if the list is full before trusting it
        if len(predictions_deque) == 10:
            most_common_pred = np.bincount(predictions_deque).argmax()
            
            # Get the confidence of the *current* frame's prediction
            # but for the *most common* class
            confidence_of_most_common = prediction[0][most_common_pred]

            # Only show if the *current frame's* confidence for the
            # *most common prediction* is high.
            if confidence_of_most_common > 0.6: # 60% confidence
                current_label = class_names[most_common_pred]
        # --- END NEW CODE ---

        # We will print the stable label
        print(f"Prediction: {current_label}")
        
    except (zmq.ZMQError, KeyboardInterrupt):
        break

socket.close()
context.term()
print("Receiver stopped.")