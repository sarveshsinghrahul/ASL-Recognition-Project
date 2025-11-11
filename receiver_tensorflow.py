# receiver_tensorflow.py
# RUN THIS IN YOUR "asl_gpu_backup" ENVIRONMENT
import cv2
import zmq
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. LOAD YOUR ASL MODEL ---
# !!! CHANGE 1: Load your new, correct model file
ASL_MODEL_PATH = 'model1.keras' 
try:
    asl_model = load_model(ASL_MODEL_PATH)
    print("ASL model loaded successfully!")
except Exception as e:
    print(f"Error loading ASL model: {e}")
    exit()

# --- 2. DEFINE CONSTANTS ---
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]
# !!! CHANGE 2: Match the IMG_SIZE you trained with
IMG_SIZE = (200, 200) 

# --- 3. SET UP ZMQ RECEIVER ---
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all messages
print("Receiver (TensorFlow) is running...")

current_label = ""
frame_counter = 0

while True:
    try:
        # --- 4. RECEIVE THE IMAGE ---
        image_bytes = socket.recv()
        
        # Decode the image from the raw bytes
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        hand_roi = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if hand_roi is None or hand_roi.size == 0:
            continue
            
        frame_counter += 1

        # --- 5. RUN PREDICTION (every 2nd frame for speed) ---
        if frame_counter % 2 == 0:
            # Pre-process this cropped image for *your* model
            img_for_model = cv2.resize(hand_roi, IMG_SIZE)
            img_for_model = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2RGB) # BGR -> RGB
            img_rescaled = img_for_model / 255.0
            img_batch = np.expand_dims(img_rescaled, axis=0)
            
            # Make a prediction
            prediction = asl_model.predict(img_batch, verbose=0)
            pred_index = np.argmax(prediction[0])
            confidence = prediction[0][pred_index]
            
            if confidence > 0.5: # 50% confidence threshold
                pred_label = class_names[pred_index]
                current_label = f"{pred_label} ({confidence*100:.2f}%)"
            else:
                current_label = "..."
        
        # --- 6. DISPLAY THE RESULT ---
        # Show the *received* hand image with its prediction
        cv2.putText(
            hand_roi, current_label, (10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.imshow('TensorFlow Receiver - Press "q" to quit', hand_roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except (zmq.ZMQError, KeyboardInterrupt):
        break

# --- 7. CLEAN UP ---
print("Cleaning up and closing...")
cv2.destroyAllWindows()
socket.close()
context.term()