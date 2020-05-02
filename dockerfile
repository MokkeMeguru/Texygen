FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN mkdir workspace
ADD requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN python -c "import nltk;  nltk.download('punkt')"
WORKDIR workspace
