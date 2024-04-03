from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load your trained model
MODEL_PATH = 'CoffeeBean_model.h5'
model = load_model(MODEL_PATH)

# Assume these are your categories based on the model's training
CATEGORIES = ['Dark', 'Green', 'Light', 'Medium']

def model_predict(img_path, model):
    IMG_SIZE = 50  # Target size for the model
    try:
        # Load the image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Resize the image
        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Normalize the image data to 0-1
        normalized_img = resized_img / 255.0
        # Reshape the image data for the model
        x = np.expand_dims(normalized_img, axis=0)
        preds = model.predict(x)
        return preds, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return jsonify({"message": 'Hello world'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
        return jsonify({"error": "Invalid file format. Only jpg, jpeg, and png files are allowed."}), 400
    
    basepath = os.path.dirname(__file__)
    temp_folder = 'temp_uploads'
    os.makedirs(os.path.join(basepath, temp_folder), exist_ok=True)
    file_path = os.path.join(basepath, temp_folder, secure_filename(file.filename))
    file.save(file_path)

    preds, error = model_predict(file_path, model)
    os.remove(file_path)  # Clean up after prediction
    
    if preds is None:
        return jsonify({"error": f"Prediction failed: {error}"}), 500

    # Process the prediction to get the category with the highest probability
    prediction_index = np.argmax(preds, axis=1)[0]
    result = CATEGORIES[prediction_index]

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

# http://127.0.0.1:5000/predict
