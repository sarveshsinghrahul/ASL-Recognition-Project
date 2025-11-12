# sender_landmark.py
import cv2
import mediapipe as mp
import zmq
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.5
)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")
print("Sender (Landmark) is running...")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # --- NORMALIZE LANDMARKS (same as in dataset creation) ---
        wrist = hand_landmarks.landmark[0]
        landmark_vector = []
        for landmark in hand_landmarks.landmark:
            landmark_vector.append(landmark.x - wrist.x)
            landmark_vector.append(landmark.y - wrist.y)
            landmark_vector.append(landmark.z - wrist.z)
        
        # --- SEND LANDMARK VECTOR (as 64-bit floats) ---
        socket.send(np.array(landmark_vector, dtype=np.float64).tobytes())
        
        # Draw skeleton on preview
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Sender - Press "q" to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
socket.close()
context.term()