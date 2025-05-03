import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

# Flask app setup
app = Flask(__name__)

# Path to the saved model
MODEL_PATH = 'models\cotton_classifier.h5'

# Load the trained model
model = load_model(MODEL_PATH)

# Label mapping
LABELS = ['Class1', 'Class2', 'Class3']  # Replace with actual class names from your dataset

# Allowed extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predict function
def predict_image(image_path):
    IMG_SIZE = (224, 224)  # Image size used during model training

    # Load the image, resize it, and preprocess
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class = LABELS[predicted_class_index]
    confidence = predictions[predicted_class_index]

    return predicted_class, confidence

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)  # Ensure uploads folder exists
        file.save(file_path)

        # Predict using the model
        predicted_class, confidence = predict_image(file_path)

        # Remove the file after prediction
        os.remove(file_path)

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}"
        })
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)