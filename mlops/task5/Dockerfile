FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["main.py"]
