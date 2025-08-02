import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model  # type: ignore

# Load model and labels
model = load_model("asl_model.h5")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # ROI coordinates
    x, y, w, h = 100, 150, 300, 300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        roi = frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (32, 32))
        roi_normalized = roi_resized / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)

        pred = model.predict(roi_input)
        class_id = np.argmax(pred)
        letter = labels[class_id]


        cv2.putText(frame, f"Letter: {letter}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No Hand Detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow("ASL Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()