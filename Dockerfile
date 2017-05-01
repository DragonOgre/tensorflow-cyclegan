FROM tensorflow/tensorflow:1.1.0-gpu-py3
RUN pip install -r requirements.txt
COPY ./ /app

