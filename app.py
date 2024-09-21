from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained CNN model
model = load_model('C:\project\plant disease detection\plant_disease_model.h5')  # Replace with your model file

@app.route('/')
def index():
    return render_template('index.html',template_folder='templates')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected image file'})

    try:
        # Read and preprocess the uploaded image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # Normalize pixel values to the range [0, 1]

        # Make a prediction using your model
        prediction = model.predict(np.array([img]))
        predicted_class = np.argmax(prediction)

        # Map class indices to class names (replace with your own mapping)
        class_names = {0: 'Tomato_Early_blight', 1: 'Tomato_Late_blight', 2: 'Tomato_healthy'}

        result = {'prediction': class_names[predicted_class]}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
