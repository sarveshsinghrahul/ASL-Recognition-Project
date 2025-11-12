# sender_mediapipe.py
# RUN THIS IN YOUR "env_mediapipe" ENVIRONMENT
import cv2
import mediapipe as mp
import zmq
import numpy as np

# --- 1. SET UP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- 2. SET UP ZMQ SENDER ---
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555") # Bind to port 5555
print("Sender (MediaPipe) is running...")

# --- 3. SET UP WEBCAM ---
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    
    # --- 4. FIND HAND AND CROP ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))
        
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop the hand from the frame
        hand_roi = frame[y_min:y_max, x_min:x_max]

        # --- 5. SEND THE CROPPED IMAGE ---
        if hand_roi.size != 0:
            # We must encode the image to send it
            _, buffer = cv2.imencode('.jpg', hand_roi)
            # Send the raw bytes
            socket.send(buffer.tobytes())
            
            # Draw box on our local preview
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('MediaPipe Sender - Press "q" to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
socket.close()
context.term()