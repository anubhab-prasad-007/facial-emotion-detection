from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect
)

from werkzeug.utils import secure_filename

from src.predict import predict_emotion

import os
import uuid
import sqlite3
from datetime import datetime


app = Flask(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

DATABASE = "emotion_history.db"


# --------------------------------------------------
# Database
# --------------------------------------------------

def init_db():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            detected_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(emotion, confidence):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    detected_at = datetime.now().strftime(
        "%d %b %Y, %I:%M %p"
    )

    cursor.execute("""
        INSERT INTO predictions
        (emotion, confidence, detected_at)
        VALUES (?, ?, ?)
    """, (
        emotion,
        confidence,
        detected_at
    ))

    connection.commit()
    connection.close()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".", 1
        )[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# Home / Image Upload
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        if "image" not in request.files:

            return render_template(
                "index.html",
                error="No image selected."
            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(
                "index.html",
                error="Please select an image."
            )

        if not allowed_file(file.filename):

            return render_template(
                "index.html",
                error="Only JPG, JPEG and PNG images are allowed."
            )

        # Safe unique filename
        original_name = secure_filename(
            file.filename
        )

        extension = original_name.rsplit(
            ".", 1
        )[1].lower()

        unique_filename = (
            f"{uuid.uuid4().hex}.{extension}"
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_filename
        )

        file.save(filepath)

        try:

            emotion, confidence, probabilities = (
                predict_emotion(filepath)
            )

            # Save prediction
            save_prediction(
                emotion,
                confidence
            )

            return render_template(
                "index.html",
                emotion=emotion,
                confidence=round(
                    confidence,
                    2
                ),
                probabilities=probabilities,
                image=unique_filename
            )

        except Exception as e:

            if os.path.exists(filepath):
                os.remove(filepath)

            return render_template(
                "index.html",
                error=str(e)
            )

    return render_template("index.html")


# --------------------------------------------------
# Webcam
# --------------------------------------------------

@app.route("/webcam", methods=["GET", "POST"])
def webcam():

    if request.method == "POST":

        if "image" not in request.files:

            return render_template(
                "webcam.html",
                error="No image received."
            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(
                "webcam.html",
                error="No image received."
            )

        unique_filename = (
            f"{uuid.uuid4().hex}.jpg"
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_filename
        )

        file.save(filepath)

        try:

            emotion, confidence, probabilities = (
                predict_emotion(filepath)
            )

            # Save webcam prediction
            save_prediction(
                emotion,
                confidence
            )

            return render_template(
                "webcam.html",
                emotion=emotion,
                confidence=round(
                    confidence,
                    2
                ),
                probabilities=probabilities,
                image=unique_filename
            )

        except Exception as e:

            if os.path.exists(filepath):
                os.remove(filepath)

            return render_template(
                "webcam.html",
                error=str(e)
            )

    return render_template("webcam.html")


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    # Total predictions
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
    """)

    total = cursor.fetchone()["total"]

    # Emotion counts
    cursor.execute("""
        SELECT emotion, COUNT(*) AS count
        FROM predictions
        GROUP BY emotion
        ORDER BY count DESC
    """)

    emotion_counts = cursor.fetchall()

    # Recent predictions
    cursor.execute("""
        SELECT emotion, confidence, detected_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """)

    recent_predictions = cursor.fetchall()

    connection.close()

    # Find most detected emotion
    most_detected = (
        emotion_counts[0]["emotion"]
        if emotion_counts
        else "None"
    )

    return render_template(
        "dashboard.html",
        total=total,
        emotion_counts=emotion_counts,
        recent_predictions=recent_predictions,
        most_detected=most_detected
    )


# --------------------------------------------------
# Serve uploaded images
# --------------------------------------------------

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )

@app.route("/clear-history", methods=["POST"])
def clear_history():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("DELETE FROM predictions")

    connection.commit()
    connection.close()

    return redirect("/dashboard")

# --------------------------------------------------
# Start application
# --------------------------------------------------

init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)