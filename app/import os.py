import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import cv2
from flask import Flask, request, jsonify, render_template

# Step 1: Set up data directories
data_dir = '/Users/shrutipataballa/Downloads/dataset-resized'  # Replace with your dataset folder
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Step 2: Data Preprocessing
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 3: Model Creation
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Model Training
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

# Save the model
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/waste_sorting_model.h5')

# Step 5: Flask Web App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    img_path = 'uploads/' + file.filename
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file.save(img_path)

    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    class_name = list(train_generator.class_indices.keys())[class_idx]

    # Determine recyclability, compostability, or landfill
    if class_name in ['glass', 'cardboard', 'paper']:
        category = 'recyclable'
    elif class_name == 'paper':
        category = 'compostable'
    elif class_name == 'plastic':
        category = 'landfill'
    else:
        category = 'unknown'

    return jsonify({'class': class_name, 'category': category, 'confidence': float(predictions[0][class_idx])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
