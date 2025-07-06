from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

datagen= ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# 
train_generator= datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

print(train_generator.class_indices)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore

# Load the MobileNetV2 model pre-trained on ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model
base_model.trainable = False

# Create a new model on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(3,activation='softmax') # 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)


# Save in .h5 format
model.save("mobilenetv2_trashident_modelV2.h5")  
