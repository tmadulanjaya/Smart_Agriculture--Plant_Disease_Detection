from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "best_model.keras"))
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")


CLASS_NAMES_FILE = os.path.join(BASE_DIR, "class_names.json")
if os.path.exists(CLASS_NAMES_FILE):
    import json
    with open(CLASS_NAMES_FILE) as f:
        CLASS_NAMES = json.load(f)
    print(f"✅ Loaded {len(CLASS_NAMES)} class names from class_names.json")
else:
  
    print("⚠️  class_names.json not found — using verified hardcoded list")
    CLASS_NAMES = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]

IMG_SIZE = (128, 128)


def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    try:
        tensor = preprocess_image(img_bytes)
        preds = model.predict(tensor)[0]
        top5_idx = preds.argsort()[-5:][::-1]

        results = [
            {
                "label": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}",
                "confidence": float(preds[i]),
            }
            for i in top5_idx
        ]

        # Encode preview image as base64
        preview = base64.b64encode(img_bytes).decode("utf-8")
        ext = file.content_type or "image/jpeg"

        return jsonify(
            {
                "predictions": results,
                "image": f"data:{ext};base64,{preview}",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": len(CLASS_NAMES)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
