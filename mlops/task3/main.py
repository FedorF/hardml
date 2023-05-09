import time

from flask import Flask, jsonify
import requests

import config as cfg


def get_secret_number(url: str) -> int:
    for i in range(100):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['secret_number']


secret_number = get_secret_number(cfg.SECRET_URL)

app = Flask(__name__)


@app.route('/return_secret_number')
def return_secret_number():
    time.sleep(1)
    return jsonify(secret_number=secret_number)
