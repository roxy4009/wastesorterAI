import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template, redirect, url_for
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import openai
import webbrowser

openai.api_key = os.getenv("OPENAI_API_KEY")

# Trained model
model = load_model('models/waste_sorting_model.h5')

class_indices_path = 'models/class_indices.npy'
if os.path.exists(class_indices_path):
    class_indices = np.load(class_indices_path, allow_pickle=True).item()
else:
    raise FileNotFoundError("Class indices file not found. Ensure you saved it during training.")
image_size = (224, 224)

# Flask webapp
app = Flask(__name__)


# Recyclability suggestions with OpenAI
def get_recyclability_suggestion(item_class, category):

    client = openai.OpenAI()  # Make sure OPENAI_API_KEY is set in environment variables

    messages = [
        {"role": "system", "content": "You are an assistant that provides suggestions for waste management and recycling."},
        {"role": "user", "content": f"Provide a detailed suggestion for properly handling and disposing of a {item_class}. It is classified as {category}."}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "Could not retrieve a suggestion at this time. Please try again later."


# Carbon footprint data (in kg of CO2 per item)
def get_carbon_footprint(item_class):
    carbon_footprint_data = {
        "glass": 0.2,  # e.g., a glass bottle
        "plastic": 1.5,  # e.g., a plastic bag
        "cardboard": 0.5,  # e.g., a small cardboard box
        "paper": 0.3,  # e.g., a sheet of paper
        "metal": 2.0,  # e.g., a can
    }
    return carbon_footprint_data.get(item_class, 0.0)

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
    class_name = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]

    # Determine recyclability, compostability, or landfill
    if class_name in ['glass', 'cardboard', 'paper']:
        category = 'recyclable'
    elif class_name == 'paper':
        category = 'compostable'
    elif class_name == 'plastic':
        category = 'landfill'
    else:
        category = 'unknown'

    confidence = float(predictions[0][class_idx]) * 100

    # Get carbon footprint data
    carbon_footprint = get_carbon_footprint(class_name)

    # Call the LLM for suggestions
    recyclability_suggestion = get_recyclability_suggestion(class_name, category)

    return render_template(
        'index.html',
        prediction=f"Class: {class_name}, Category: {category}",
        confidence=confidence,
        img_path=img_path,
        carbon_footprint=carbon_footprint,
        suggestion=recyclability_suggestion
    )

if __name__ == '__main__':
    # Open the web browser automatically
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
