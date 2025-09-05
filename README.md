# ♻️ AI Waste Sorting Assistant

This project is a full-stack AI-powered waste sorting web app that classifies uploaded waste images into categories like recyclable, compostable, or landfill. It provides **dynamic recycling suggestions** via OpenAI and estimates **carbon footprint impact** for each item.

Built using **TensorFlow**, **OpenCV**, **Flask**, and **OpenAI GPT**, the system helps users dispose of waste responsibly using image recognition and generative AI.

This model supports classification of:
- Glass
- Plastic
- Cardboard
- Paper
- Metal  
_(Additional classes can be added by retraining the model with new images.)_

---

## Features

- ✅ **Waste classification** using a custom-trained MobileNetV2 model (TensorFlow)
- 🧠 **OpenAI-powered suggestions** for proper disposal instructions
- 🌱 **Carbon footprint estimates** for each item class
- 📷 **Image preprocessing and augmentation** for improved model performance
- 📊 **Confidence scoring** with visualization
- 🧪 Local web app built using **Flask**

---

## AI & ML Techniques Used

- **Transfer Learning** with MobileNetV2 for fast and accurate classification
- **Image Augmentation** (rotation, zoom, flip, shift) to improve generalization
- **Dropout + ReLU + Softmax Layers** for robust classification and reduced overfitting
- **Adam Optimizer** and categorical crossentropy loss for model training
- **Early stopping & validation monitoring** for optimal training cycles

---

## Running the project
Due to GitHub’s file size limits, the trained model is hosted externally.
👉 Download the model here https://drive.google.com/file/d/1nX7I3OLI_7v6Ic8clj24PSbJZgIA7MGL/view?usp=sharing&utm_source=chatgpt.com
After downloading, place the file into the models/ directory
```bash
- export OPENAI_API_KEY=your_key_here
- pip install -r requirements.txt
- python app.py
```


