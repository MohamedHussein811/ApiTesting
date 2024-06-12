from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

models = {
    "denseNetModel": None,
    "inceptionV3": None,
    "Xception": None,
    "cnnscratch": None,
    "vgg19": None,
}

model_paths = {
    "denseNetModel": "best_denseNet_model.tflite",
    "inceptionV3": "best_inception_V3_model.tflite",
    "Xception": "best_Xception_model.tflite",
    "cnnscratch": "scratch.tflite",
    "vgg19": "best_vgg16_model.tflite",
}

def load_tf_lite_models():
    global models
    for key, path in model_paths.items():
        if models[key] is None:
            print(f"Loading {key} from {path}")
            models[key] = tf.lite.Interpreter(model_path=path)
            models[key].allocate_tensors()
            print(f"{key} loaded")

def process_and_predict_image(file, model_key):
    img = Image.open(file)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize image

    input_details = models[model_key].get_input_details()
    output_details = models[model_key].get_output_details()

    input_shape = input_details[0]['shape']
    models[model_key].set_tensor(input_details[0]['index'], [img_array])

    models[model_key].invoke()
    output_data = models[model_key].get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(output_data, axis=1)
    prediction_percentages = (output_data[0] * 100).tolist()

    return predicted_class[0], prediction_percentages

@app.route('/predict/<model_key>', methods=['POST'])
def predict(model_key):
    file = request.files['file']
    predicted_class, prediction_percentages = process_and_predict_image(file, model_key)
    return jsonify({
        'predicted_class': int(predicted_class),
        'prediction_percentages': prediction_percentages
    })

if __name__ == '__main__':
    load_tf_lite_models()
    app.run()

# Add the line below for Vercel to recognize the app
app = app
