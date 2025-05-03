from flask import Blueprint, request, jsonify, render_template
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Create Blueprint
jari_app = Blueprint('jari_app', __name__, template_folder='templates')

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'zari_classifier_model.h5')
model = load_model(MODEL_PATH)

# Define labels
LABELS = {0: "silver jari", 1: "gold jari"}  # Removed 'not jari'

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    try:
        # Preprocess the image
        img = load_img(image_path, target_size=(150, 150))  # Ensure this matches your training input size
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = LABELS.get(predicted_class, "Unknown")
        return label
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction"

# Routes
@jari_app.route('/jari', methods=['GET'])
def jari_home():
    return render_template('index.html')

@jari_app.route('/jari/predict', methods=['POST'])
def jari_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Get prediction
        result = predict_image(file_path)
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({'prediction': result})
    else:
        return jsonify({'error': 'Invalid file type'}), 400