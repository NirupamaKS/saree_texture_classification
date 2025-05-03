from flask import Blueprint, request, jsonify, render_template
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

silk_app = Blueprint('silk_app', __name__, template_folder='templates')

# Update the model path if necessary
MODEL_PATH = r'C:/Users/Lenovo/Desktop/PROJECTS/cotton/models/silk_saree_classifier_improved.h5'  
model = load_model(MODEL_PATH)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class labels corresponding to your model output
CLASS_LABELS = ['Mixed Silk','Not Silk', 'Pure Silk']

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Preprocess the image and make predictions."""
    # Load image with correct target size (224x224 as per the model input)
    image = load_img(image_path, target_size=(224, 224))  # Ensure it's 224x224
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image
    
    # Get model predictions
    predictions = model.predict(image)[0]
    
    # Get the predicted class index and the associated confidence
    predicted_index = np.argmax(predictions)
    
    return CLASS_LABELS[predicted_index], predictions[predicted_index]

@silk_app.route('/silk', methods=['GET'])
def silk_home():
    """Render the home page."""
    return render_template('index.html')

@silk_app.route('/silk/predict', methods=['POST'])
def silk_predict():
    """Handle image prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400  # No file uploaded
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400  # No file selected
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)  # Create the uploads directory if it doesn't exist
        file.save(file_path)
        
        # Make prediction
        label, confidence = predict_image(file_path)
        
        # Clean up by removing the uploaded file after prediction
        os.remove(file_path)
        
        return jsonify({
            'prediction': label,
        })
    else:
        return jsonify({'error': 'Invalid file type'}), 400  # Invalid file extension
