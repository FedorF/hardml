import time

from flask import Flask, jsonify

import config as cfg


app = Flask(__name__)


@app.route('/return_secret_number')
def return_secret_number():
    time.sleep(1)
    return jsonify(secret_number=cfg.SECRET_NUMBER)


if __name__ == "__main__":
    app.run(host=cfg.FLASK_HOST, port=cfg.FLASK_PORT)
