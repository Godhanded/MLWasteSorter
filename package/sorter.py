import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

def get_resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

cap = cv2.VideoCapture(0)

model = load_model("./main_mobilenetv2_trashident_modelV2.h5") # Load the saved model

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess image
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # Predict
    pred = model.predict(img)
    label = ['Food Waste', 'Recycling', 'Trash'][np.argmax(pred)]
    # Display result
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()