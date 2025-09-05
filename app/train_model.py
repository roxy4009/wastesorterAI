import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

# Paths
data_dir = '/Users/shrutipataballa/Downloads/dataset-resized'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
model_dir = 'models'

# Preprocessing
image_size = (224, 224)
batch_size = 32

# Augemnting only training data, reduce overfitting 
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

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

# Model creation, using MobileNetV2 CNN
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False # remove default clasification layer to create custom layer

# Adding cutom layer
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x) # ReLU for non-linearity, complex patterns, vanishing grad problem
x = Dropout(0.5)(x) # Dropout layer, max 0.5
output_layer = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Compiling model, adam optimizer for adaptive learning 
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training, stops if no validation loss improvement for 5 epochs
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save(os.path.join(model_dir, 'waste_sorting_model.h5'))

# Saveing class indices
np.save(os.path.join(model_dir, 'class_indices.npy'), train_generator.class_indices)
# print("Class indices saved successfully.")

print("Model and class indices saved successfully.")
