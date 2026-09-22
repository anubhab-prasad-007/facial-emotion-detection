import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/emotion_model.keras")

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def predict_emotion(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # Grayscale image for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        raise ValueError(
            "No face detected. Please upload a clear face image."
        )

    # Select largest face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Crop face BEFORE drawing the rectangle
    face = gray[y:y + h, x:x + w]

    # Resize to CNN input
    face = cv2.resize(face, (48, 48))

    # Normalize
    face = face.astype("float32") / 255.0

    # Reshape
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    # Prediction
    prediction = model.predict(face, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    emotion = emotion_labels[predicted_index]

    confidence = float(
        prediction[predicted_index] * 100
    )

    # All probabilities
    probabilities = {
        label: float(probability * 100)
        for label, probability in zip(
            emotion_labels,
            prediction
        )
    }

    # Draw face rectangle
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    # Display emotion above face
    label = f"{emotion} {confidence:.1f}%"

    cv2.putText(
        image,
        label,
        (x, max(y - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Save annotated image
    cv2.imwrite(image_path, image)

    return emotion, confidence, probabilities