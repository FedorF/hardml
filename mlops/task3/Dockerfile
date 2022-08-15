FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app
RUN apt update
RUN apt -y install gcc
RUN python3 -m pip install -r requirements.txt

CMD uwsgi --http 0.0.0.0:9000 --wsgi-file main.py --callable app --lazy-apps --master-fifo=/app/master.fifo --processes 2 --threads 5
