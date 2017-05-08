from prediction import get_predictors
from flask import Flask, request
import base64
import numpy as np
import io
from PIL import Image
import scipy.misc
import json

def get_predict_func(predictors):
    def predict(predictor_name='sunny_sunset/sunset'):
        print(request)
        incoming = request.get_json()
        image_data = incoming['imageData']
        image_data = image_data.replace('data:image/jpeg;base64,','')
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
        decoded_image = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
        predictor = predictors[predictor_name]
        result_image = predictor(decoded_image)
        result_image_bytes = result_image.tobytes()
        encoded_result_image = base64.b64encode(result_image_bytes)
        return json.dumps({'imageData': encoded_result_image.decode('utf-8')}), 200, {'ContentType':'application/json'}

    return predict

def hello_world():
    return 'Hello, World!'

def main():
    predictors = get_predictors()
    app = Flask(__name__)
    app.add_url_rule('/', view_func=hello_world)
    app.add_url_rule('/predict', view_func=get_predict_func(predictors), methods=['POST'])
    app.run(host= '0.0.0.0', port=80)


if __name__ == '__main__':
    main()

# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello, World!'