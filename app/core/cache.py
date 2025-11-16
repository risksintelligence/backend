import os
import redis

class RedisCache:
    def __init__(self, namespace: str) -> None:
        url = os.getenv('RRIO_REDIS_URL', 'redis://localhost:6379/0')
        self.client = redis.from_url(url)
        self.namespace = namespace

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str):
        value = self.client.get(self._key(key))
        if value:
            return value
        return None

    def set(self, key: str, value, ttl: int = 900):
        self.client.set(self._key(key), value, ex=ttl)
