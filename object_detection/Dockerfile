FROM ubuntu:18.04
FROM python:3
FROM tensorflow/tensorflow:2.2.2-gpu

ADD od_app /od_app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR "./od_app"
CMD ["python", "./app.py"]
