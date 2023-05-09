from flask import Flask, jsonify
import requests

import config as cfg
import registry


def get_secret_number(url: str) -> int:
    for i in range(100):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['secret_number']


secret_number = get_secret_number(cfg.SECRET_URL)

app = Flask(__name__)


@app.route('/return_secret_number')
def return_secret_number():
    return jsonify(secret_number=secret_number)


if __name__ == "__main__":
    r = registry.RedisRegistry(
        host=cfg.REDIS_HOST,
        port=cfg.REDIS_PORT,
        password=cfg.PASSWORD,
        service_name=cfg.SERVICE_NAME,
    )
    r.register(
        replica_name=cfg.REPLICA_NAME,
        properties=cfg.PROPERTIES,
    )
    app.run(host=cfg.FLASK_HOST, port=cfg.FLASK_PORT)
