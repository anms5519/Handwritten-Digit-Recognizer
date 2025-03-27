from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
model = load_model('mnist_model.h5')

def preprocess_image(image_path):
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to 28x28 pixels to match MNIST images
    img = cv2.resize(img, (28, 28))
    # Normalize pixel values and reshape for the model
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Save the uploaded file
        file = request.files['image']
        filepath = os.path.join('static', 'uploaded.jpg')
        os.makedirs('static', exist_ok=True)
        file.save(filepath)
        
        # Process the image and predict
        img = preprocess_image(filepath)
        pred = model.predict(img)
        prediction = np.argmax(pred)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
