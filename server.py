from prediction import get_predictors
from flask import Flask, request
import base64
import numpy as np
import io
from PIL import Image
import scipy.misc
import json
from flask_cors import CORS, cross_origin

def get_predict_func(predictors):
    def predict():
        incoming = request.get_json()
        image_data = incoming['imageData']
        predictor_name = incoming['filter']
        print('New request for filter {}'.format(predictor_name))
        image_data = image_data.replace('data:image/jpeg;base64,','')
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))  

        half_the_width = image.size[0] / 2
        half_the_height = image.size[1] / 2
        cropped_image = image.crop(
            (
                half_the_width - 128,
                half_the_height - 128,
                half_the_width + 128,
                half_the_height + 128
            )
        )
        
        decoded_image = np.array(cropped_image.getdata()).reshape(256, 256, 3)

        predictor = predictors[predictor_name]
        result_image = predictor(decoded_image)
        
        img = scipy.misc.toimage(result_image)
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()

        encoded_string = base64.b64encode(imgByteArr).decode()

        return json.dumps({'imageData': encoded_string}), 200, {'ContentType':'application/json'}

    return predict

def hello_world():
    return 'Hello, World!'

def main():
    predictors = get_predictors()
    app = Flask(__name__)
    CORS(app)
    app.add_url_rule('/', view_func=hello_world)
    app.add_url_rule('/predict', view_func=get_predict_func(predictors), methods=['POST', 'OPTIONS'])
    app.run(host= '0.0.0.0', port=80)


if __name__ == '__main__':
    main()
