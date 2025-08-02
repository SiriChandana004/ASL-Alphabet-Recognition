import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load model and labels
model = load_model('asl_model.h5')
labels = sorted(os.listdir('dataset/asl_alphabet_train'))  # Dynamic labels

# Webcam
cap = cv2.VideoCapture(0)
img_size = 32

while True:
    success, frame = cap.read()
    if not success:
        break

    # ROI box
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    roi_resized = cv2.resize(roi, (img_size, img_size))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = roi_normalized.reshape(1, img_size, img_size, 3)

    prediction = model.predict(roi_reshaped)
    class_id = np.argmax(prediction)
    label = labels[class_id]

    # Display prediction
    cv2.putText(frame, f"Predicted: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()