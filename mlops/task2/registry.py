import redis


class RedisRegistry:
    def __init__(self, host: str, port: str, password: str, service_name: str):
        super().__init__()
        self.host = host
        self.port = port
        self.password = password
        self.service_name = service_name

    def _connect_to_redis(self) -> redis.StrictRedis:
        registry = redis.StrictRedis(
            host=self.host,
            port=self.port,
            password=self.password,
            decode_responses=True,
        )
        return registry

    def register(self, replica_name: str, properties: dict):
        with self._connect_to_redis() as registry:
            registry.lpush(self.service_name, replica_name)
            registry.hmset(replica_name, properties)
