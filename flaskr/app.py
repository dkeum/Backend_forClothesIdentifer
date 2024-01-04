import os

from flask import Flask
from flask_cors import CORS,  cross_origin
from flask import request, jsonify
import torch
from model import FashionMNISTModelV2
from predict import make_predictions
from PIL import Image
import numpy as np

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/hello', methods=['GET'])
@cross_origin()
def hello():
    response_data = {"message": "hi"}
    return jsonify(response_data), 200

def predict_classes():

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Convert the uploaded image to PIL format and resize it
        image_pil = Image.open(file).convert('L')  # Convert image to grayscale
        image_resized = image_pil.resize((28, 28))  # Resize image to 28x28
        
        # image_resized.show()

        # Convert image to numpy array and normalize pixel values
        image_array = np.array(image_resized) / 255.0

        # Reshape the image to match the model's expected input shape
        image_reshaped = image_array.reshape(1, 28, 28)

        # turn ndarray into tensor
        image_reshaped = torch.from_numpy(image_reshaped)

        # change from double to float
        image_reshaped = [image_reshaped.float()]

        # Load your model (FashionMNISTModelV2) and make predictions
        loaded_model = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=10)
        MODEL_SAVE_PATH = 'flaskr/models/cv_model.pth'
        loaded_state_dict = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))
        loaded_model.load_state_dict(loaded_state_dict)

        # Make predictions
        pred_probs = make_predictions(model=loaded_model, data=image_reshaped)
        # print(pred_probs)
        pred_classes = pred_probs.argmax(dim=1).numpy()[0]
        # print(pred_classes)
        labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        prediction = labels[pred_classes]

        response_data = {"prediction": str(prediction)}
        return jsonify(response_data), 200



# def create_app(test_config=None):
#     # create and configure the app
#     app = Flask(__name__, instance_relative_config=True)
#     CORS(app)
#     cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
#     app.config.from_mapping(
#         SECRET_KEY='dev',
#         DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
#     )


#     if test_config is None:
#         # load the instance config, if it exists, when not testing
#         app.config.from_pyfile('config.py', silent=True)
#     else:
#         # load the test config if passed in
#         app.config.from_mapping(test_config)

#     # ensure the instance folder exists
#     try:
#         os.makedirs(app.instance_path)
#     except OSError:
#         pass
#     @app.route('/', methods=['GET'])
#     @cross_origin()
#     def default():
#         # response_data = {"message": "hi"}
#         # return jsonify(response_data), 200
#         return " default path"
    
#     @app.route('/hello', methods=['GET'])
#     @cross_origin()
#     def hello():
#         response_data = {"message": "hi"}
#         return 'HI'
#         # return jsonify(response_data), 200
#     # @app.route('/api/predict', methods=['POST'])
#     # @cross_origin()
#     # def predict_classes():

#     #     if request.method == 'POST':
#     #         if 'file' not in request.files:
#     #             return jsonify({'error': 'No file part'})

#     #         file = request.files['file']
  
#     #         # Convert the uploaded image to PIL format and resize it
#     #         image_pil = Image.open(file).convert('L')  # Convert image to grayscale
#     #         image_resized = image_pil.resize((28, 28))  # Resize image to 28x28
            
#     #         # image_resized.show()

#     #         # Convert image to numpy array and normalize pixel values
#     #         image_array = np.array(image_resized) / 255.0

#     #         # Reshape the image to match the model's expected input shape
#     #         image_reshaped = image_array.reshape(1, 28, 28)

#     #         # turn ndarray into tensor
#     #         image_reshaped = torch.from_numpy(image_reshaped)

#     #         # change from double to float
#     #         image_reshaped = [image_reshaped.float()]

#     #         # Load your model (FashionMNISTModelV2) and make predictions
#     #         loaded_model = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=10)
#     #         MODEL_SAVE_PATH = 'flaskr/models/cv_model.pth'
#     #         loaded_state_dict = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))
#     #         loaded_model.load_state_dict(loaded_state_dict)

#     #         # Make predictions
#     #         pred_probs = make_predictions(model=loaded_model, data=image_reshaped)
#     #         # print(pred_probs)
#     #         pred_classes = pred_probs.argmax(dim=1).numpy()[0]
#     #         # print(pred_classes)
#     #         labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#     #         prediction = labels[pred_classes]

#     #         response_data = {"prediction": str(prediction)}
#     #         return jsonify(response_data), 200

        

#     return app

