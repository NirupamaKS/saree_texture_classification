from flask import Blueprint, request, jsonify, render_template
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Create Blueprint
cotton_app = Blueprint('cotton_app', __name__, template_folder='templates')

# Load the model
MODEL_PATH =r'C:\Users\Lenovo\Desktop\PROJECTS\cotton\models\cotton_classification_model.h5'

model = load_model(MODEL_PATH)

# Define labels
LABELS = {1: "pure_cotton", 0: "mixed_cotton"}  # Two classes: pure_cotton and mixed_cotton

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
@cotton_app.route('/cotton', methods=['GET'])
def cotton_home():
    return render_template('index.html')

@cotton_app.route('/cotton/predict', methods=['POST'])
def cotton_predict():
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