import tkinter as tk
from tkinter import Label, Button
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image, ImageTk

# Load trained model and label list
model = load_model("asl_model.h5")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# MediaPipe hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# GUI setup
root = tk.Tk()
root.title("Gesture Genie")
root.attributes('-fullscreen', True)
root.configure(bg="#f2f2f2")

# Load and set background image
bg_img = Image.open("bckgrnd.jpg")
bg_img = bg_img.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
bg_photo = ImageTk.PhotoImage(bg_img)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# --- Title & Tagline ---
title_label = tk.Label(
    root,
    text="GESTURE GENIE",
    font=("Segoe UI", 48, "italic"),
    bg="#f2f2f2",
    fg="#8A2BE2"
)
title_label.pack(pady=(30, 10))

tagline_label = tk.Label(
    root,
    text="Where Gestures Become Letters," \
    "and Letters Become Voice....",
    font=("Arial", 20, "italic"),
    bg="#f2f2f2",
    fg="#8A2BE2"
)
tagline_label.pack(pady=(0, 20))

# Webcam
cap = None

# Label for prediction
prediction_label = Label(root, text="Prediction: ", font=("Arial", 24), bg="#f2f2f2", fg="#007acc")
prediction_label.pack(pady=10)

# Label for video feed
video_label = Label(root)
video_label.pack()

# Frame for buttons
button_frame = tk.Frame(root, bg="#f2f2f2")
button_frame.pack(pady=30)

# Buttons (placeholders)
start_button = None
exit_button = None

# Video frame update function
def update_frame():
    global cap
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    x, y, w, h = 100, 150, 300, 300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        roi = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (32, 32))
        roi_normalized = roi_resized / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)

        pred = model.predict(roi_input)
        class_id = np.argmax(pred)
        letter = labels[class_id]

        prediction_label.config(text=f"Prediction: {letter}")
    else:
        prediction_label.config(text="Prediction: -")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_frame)

# Start the webcam
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()
    start_button.pack_forget()
    exit_button.pack()

# Stop everything
def stop():
    if cap:
        cap.release()
    root.destroy()

# Buttons
start_button = Button(button_frame, text="Start Recognition", command=start_camera, font=("Arial", 20), bg="green", fg="white", padx=30, pady=10)
start_button.pack(side=tk.LEFT, padx=20)

exit_button = Button(button_frame, text="Exit", command=stop, font=("Arial", 20), bg="red", fg="white", padx=30, pady=10)
exit_button.pack(side=tk.LEFT, padx=20)
exit_button.pack_forget()

root.mainloop()