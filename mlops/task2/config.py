import os

REDIS_HOST = '135.181.204.59'
REDIS_PORT = '6379'
FLASK_HOST = '0.0.0.0'
FLASK_PORT = '8000'
PASSWORD = 'babushka'
SERVICE_NAME = 'web_app'
REPLICA_NAME = os.environ['REPLICA_NAME']
PROPERTIES = {'host': REDIS_HOST, 'port': os.environ['REPLICA_PORT']}
SECRET_URL = 'https://lab.karpov.courses/hardml-api/module-5/get_secret_number'
