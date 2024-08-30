from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import joblib
import cv2
from skimage.transform import resize
from skimage.feature import hog

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('cnn_model.h5')
clf = joblib.load('rf_model.pkl')

def process_image_cnn(file_storage):
    img = image.load_img(file_storage, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def preprocess_image_rf(image):
    image = resize(image, (100, 100))
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
    return features

class_labels = {
    0: 'scab',
    1: 'Black rot',
    2: 'Cedar rust',
    3: 'healthy',
    4: 'healthy',
    5: 'Powdery mildew',
    6: 'healthy',
    7: 'Cercospora_leaf_spot',
    8: 'Common rust',
    9: 'Northern_Leaf_Blight',
    10: 'healthy',
    11:'Black_rot',
    12:'Esca (Black_Measles)',
    13:'Leaf blight (Isariopsis_Leaf_Spot)', 
    14:'healthy', 
    15:'Haunglongbing (Citrus_greening)',
    16:'Bacterial spot',
    17:'healthy',
    18:'Bacterial spot',
    19:'healthy',
    20:'Early blight',
    21:'Late blight',
    22:'healthy',
    23:'healthy',
    24:'healthy',
    25:'Powdery mildew',
    26:'Leaf scorch',
    27:'healthy',
    28:'Bacterial spot',
    29:'Early blight',
    30:'Late blight',
    31:'Leaf Mold',
    32:'Septoria leaf spot',
    33:'Spider_mites ',
    34:'Target Spot',
    35:'Yellow Leaf Curl Virus', 
    36:'mosaic virus',
    37:'healthy'
    # Define your class labels here
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    image_data = file.read()
    if not image_data:
        return jsonify({'error': 'Failed to read image data'}), 400
    
    try:
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        processed_image = preprocess_image_rf(image)
        processed_image = processed_image.reshape(1, -1)
        prediction = clf.predict(processed_image)
        predicted_class_rf = str(prediction[0])
        #print("Random Forest Prediction:", predicted_class_rf)
        return jsonify({'predicted_class_rf': predicted_class_rf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    file = request.files['file']
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        img_array = process_image_cnn(filename)
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction)
        predicted_class_label = class_labels.get(predicted_class_idx, "Unknown")
        return jsonify({'predicted_class_cnn': predicted_class_label})
    return jsonify({'error': 'No file uploaded'})



if __name__ == "__main__":
    app.run(debug=True)


