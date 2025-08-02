import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# === CONFIGURATION ===
img_size = 32
batch_size = 32
data_dir = "dataset/asl_alphabet_train"  # 👈 update this if needed
model_path = "asl_model.h5"

# === LOAD MODEL ===
model = load_model(model_path)
print("✅ Model loaded.")

# === SETUP DATA GENERATOR (Only for Evaluation - No Augmentation) ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(32,32),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False 
)

labels=list(val_generator.class_indices.keys())
# === PREDICT ===
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
labels = list(val_generator.class_indices.keys())

# === REPORT ===
print("📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
print("Class indices:",val_generator.class_indices)