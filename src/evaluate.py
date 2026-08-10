import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = tf.keras.models.load_model("models/emotion_model.keras")

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/fer2013.csv/test",
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False
)

# Normalize images
test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))

# Predict
y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Emotion labels
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))