

class RedisConnection:

    def __init__(self):
        pass

    def _connect(self):
        pass


class RedisCache(RedisConnection):

    def __init__(self):
        super().__init__()

    def put_data_into_cache(self):
        pass

    def get_data_from_cache(self):
        pass


class RedisStream(RedisConnection):

    def __init__(self):
        super().__init__()

    def put_into_stream(self):
        pass


def get_key(placeholder: dict):
    pass