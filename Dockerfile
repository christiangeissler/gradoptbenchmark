FROM continuumio/anaconda3:latest

WORKDIR /app

ADD requirements.txt /app

RUN apt-get update
RUN apt-get -y install -y python3-tk
RUN apt-get -y install build-essential swig cmake
RUN curl https://raw.githubusercontent.com/automl/smac3/master/requirements.txt | xargs -n 1 -L 1 pip install 

RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "Experiment.py"]