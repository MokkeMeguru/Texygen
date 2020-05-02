FROM tensorflow/tensorflow:1.5.0-gpu-py3
RUN mkdir workspace
ADD requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt
WORKDIR workspace
