# Waste Classifier Web Application

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
Flask==2.3.3
Flask-SocketIO==5.3.6
tensorflow==2.13.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
python-socketio==5.9.0
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. File Structure
The project has this structure:
```
waste_classifier_main/
├── app.py                    # Flask backend (from the code above)
├── templates/
│   └── index.html           # Frontend HTML (from the artifact)
├── main_mobilenetv2_trashident_modelV2.h5  # The trained model
└── requirements.txt
```

### 3. Save the Files

**Save the Flask backend as `app.py`:**
- Copy the Flask backend code from the artifact above

**Create the templates directory and save the HTML:**
```bash
mkdir templates
# Save the HTML artifact as templates/index.html
```

**Make sure the model file is in the same directory:**
- Place `main_mobilenetv2_trashident_modelV2.h5` in the root directory

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Application
- Open your browser and go to `http://localhost:5000`
- Allow camera permissions when prompted
- The application will automatically start classifying waste items in real-time

## Features

✅ **Real-time Classification**: Uses the trained TensorFlow model  
✅ **Modern UI**: Beautiful glassmorphism design  
✅ **WebSocket Communication**: Real-time updates without page refresh  
✅ **Statistics Tracking**: Tracks classification counts and accuracy  
✅ **Responsive Design**: Works on desktop and mobile  
✅ **Error Handling**: Graceful error handling for camera and model issues  

## How It Works

1. **Frontend**: Captures video frames from the camera and sends them to the backend
2. **Backend**: Processes frames using the TensorFlow model and returns predictions
3. **WebSocket**: Real-time communication between frontend and backend
4. **Display**: Results are displayed with confidence scores and visual indicators

## Troubleshooting

**Model not loading:**
- Ensure `main_mobilenetv2_trashident_modelV2.h5` is in the same directory as `app.py`
- Check that TensorFlow is installed correctly

**Camera not working:**
- Make sure to allow camera permissions in your browser
- Try using HTTPS if on a remote server (cameras require secure context)

**Connection issues:**
- Check that Flask is running on the correct port
- Ensure no firewall is blocking the connection

## Production Deployment

For production deployment, consider:
- Using a production WSGI server like Gunicorn
- Adding HTTPS support
- Implementing user authentication if needed
- Adding rate limiting for the classification endpoint
- Optimizing model loading and inference speed

## Performance Notes

- Classifications are throttled to 1 per second to avoid overloading
- Images are compressed to JPEG format before sending to reduce bandwidth
- The model is loaded once at startup for better performance

**Link to images used for training**: https://www.kaggle.com/datasets/mostafaabla/garbage-classification