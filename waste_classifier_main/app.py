import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import threading
import time
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecrete key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
model = None
classification_active = False
last_classification_time = 0

def get_resource_path(relative_path):
    """Get resource path for PyInstaller compatibility"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_classification_model():
    """Load the TensorFlow model"""
    global model
    try:
        model_path = get_resource_path("main_mobilenetv2_trashident_modelV2.h5")
        if not os.path.exists(model_path):
            # Try current directory if resource path doesn't work
            model_path = "main_mobilenetv2_trashident_modelV2.h5"
        
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data URL prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Normalize
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image(image_data):
    """Classify image using the loaded model"""
    global model, last_classification_time
    
    if model is None:
        return None
    
    # Throttle classifications to avoid overloading
    current_time = time.time()
    if current_time - last_classification_time < 1.0:  # Limit to 1 classification per second
        return None
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return None
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get class labels
        labels = ['Food Waste', 'Recycling', 'Trash']
        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # Get all class probabilities
        class_probabilities = {
            labels[i]: float(prediction[0][i]) for i in range(len(labels))
        }
        
        last_classification_time = current_time
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities
        }
    except Exception as e:
        print(f"Error during classification: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to waste classifier server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_classification')
def handle_start_classification():
    """Handle start classification request"""
    global classification_active
    classification_active = True
    emit('classification_status', {'active': True})
    print('Classification started')

@socketio.on('stop_classification')
def handle_stop_classification():
    """Handle stop classification request"""
    global classification_active
    classification_active = False
    emit('classification_status', {'active': False})
    print('Classification stopped')

@socketio.on('classify_frame')
def handle_classify_frame(data):
    """Handle frame classification request"""
    global classification_active
    
    if not classification_active:
        return
    
    if model is None:
        emit('classification_result', {
            'error': 'Model not loaded',
            'success': False
        })
        return
    
    try:
        image_data = data.get('image')
        if not image_data:
            return
        
        # Classify the image
        result = classify_image(image_data)
        
        if result:
            emit('classification_result', {
                'success': True,
                'class': result['class'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            })
        else:
            emit('classification_result', {
                'success': False,
                'error': 'Classification failed'
            })
    except Exception as e:
        print(f"Error in classify_frame: {e}")
        emit('classification_result', {
            'success': False,
            'error': str(e)
        })

@socketio.on('get_model_status')
def handle_get_model_status():
    """Handle model status request"""
    global model
    status = {
        'loaded': model is not None,
        'ready': model is not None
    }
    emit('model_status', status)

if __name__ == '__main__':
    # Load the model on startup
    print("Loading waste classification model...")
    model_loaded = load_classification_model()
    
    if model_loaded:
        print("Model loaded successfully!")
    else:
        print("Warning: Model could not be loaded. Classification will not work.")
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    
    # Run the Flask-SocketIO server
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)