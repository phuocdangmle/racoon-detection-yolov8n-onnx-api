FROM python:3.8

RUN mkdir /app

WORKDIR /app/src
COPY . /app/src

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

CMD ["python", "-u", "app.py"]