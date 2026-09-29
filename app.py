import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ============================================
# Load NeuroVision Model
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "neurovision_resnet50.keras")

model = load_model(MODEL_PATH)

# ============================================
# Class Mapping
# ============================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor"
]

# ============================================
# Image Preprocessing
# ============================================

def prepare_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Do NOT divide by 255
    # Model was trained using 0-255 input values

    return img_array


# ============================================
# Home Page
# ============================================

@app.route('/')
def home():
    return render_template('index.html')


# ============================================
# Prediction
# ============================================

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({
            'error': 'No file uploaded'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'error': 'Empty filename'
        }), 400

    # Create upload folder
    upload_folder = os.path.join(
        BASE_DIR,
        'static',
        'uploads'
    )

    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(
        upload_folder,
        file.filename
    )

    file.save(file_path)

    try:

        # Prepare image
        processed_img = prepare_image(file_path)

        # Prediction
        predictions = model.predict(
            processed_img,
            verbose=0
        )

        # Find predicted class
        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_index]
        ) * 100

        return jsonify({

            'prediction':
                CLASS_NAMES[predicted_index],

            'confidence':
                f"{confidence:.2f}%",

            'image_url':
                f"/static/uploads/{file.filename}"
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500


# ============================================
# Run Flask Application
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)