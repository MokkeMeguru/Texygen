FROM tensorflow/tensorflow:1.15.2-gpu-py3
RUN apt update
RUN apt install -y tmux wget git
RUN mkdir workspace
ADD requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN python -c "import nltk;  nltk.download('punkt')"
WORKDIR workspace
