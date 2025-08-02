import cv2
import os
import sys

# ----------- SETTINGS -----------
label = input("Enter the letter you want to capture (e.g., A): ").upper()
save_dir = f"dataset/asl_alphabet_train/{label}"
os.makedirs(save_dir, exist_ok=True)

img_size = 32
capture_count = 0
total_images = 30  # Change as needed

# ----------- START CAMERA -----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    sys.exit()

print("\n📸 Press 's' to save, 'q' to quit.\n")

# ----------- LOOP -----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    # Define ROI
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (img_size, img_size))

    # Show window
    cv2.imshow("Capture Hand Sign", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save image
    if key == ord('s'):
        filename = os.path.join(save_dir, f"{label}_{capture_count}.jpg")
        cv2.imwrite(filename, roi_resized)
        capture_count += 1
        print(f"[{capture_count}/{total_images}] Saved {filename}")
        if capture_count >= total_images:
            print("✅ Done capturing.")
            break

    # Quit
    elif key == ord('q'):
        print("❌ Quit pressed. Closing...")
        break

# ----------- CLEANUP -----------
cap.release()
cv2.destroyAllWindows()
sys.exit()