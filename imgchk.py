import cv2
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore

model= load_model("mobilenetv2_trashident_model.h5")

img_path = "OIP.jpg"  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
label = ['Biological Waste', 'Recyclable', 'Trash'][np.argmax(pred)]
print(f"Predicted Class: {label}")
