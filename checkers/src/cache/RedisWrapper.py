import redis
import aioredis
import hashlib
import asyncio
import logging


class RedisChannel:

    def __init__(self, host: str = "localhost", port: int =6379, db: int = 1):
        self.host = host
        self.db = db
        self.port = port
        self.redis_cache = None

        async def _create_redis_cache():
            self.redis_cache = await aioredis.create_redis((self.host, self.port))

        tasks = [asyncio.ensure_future(_create_redis_cache())]
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(*tasks))

    def put_into_channel(self, channel, message):
        self.redis_cache.publish(channel, message)


class RedisCache:

    def __init__(self, host: str = "localhost", port: int =6379, db: int = 1):
        self.host = host
        self.port = port
        self.db = db
        self.redis_cache = redis.StrictRedis(host=host, port=port, db=db)

    def put_data_into_cache(self, key, value):
        try:
            self.redis_cache.set(key, value)
        except:
            logging.error("Value not serializable!")

    def get_data_from_cache(self, key):
        if self.redis_cache.exists(key):
            return self.redis_cache.get(key)
        else:
            raise KeyError("Key not present in redis cache")


def get_key(value: str) -> str:
    """
    creates hash key for given string
    :param value:
    :return:
    """
    return hashlib.md5(value.encode("utf-8"))
