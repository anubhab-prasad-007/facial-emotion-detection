from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout, BatchNormalization
)

def build_model():
    model = Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(32, (3,3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.5),

        Dense(7, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()