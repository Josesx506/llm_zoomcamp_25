from datetime import datetime, timedelta
from functools import lru_cache, wraps


def timed_lru_cache(seconds: int, maxsize: int = None, verbose=False):
    """
    Author: https://www.mybluelinux.com/pyhon-lru-cache-with-time-expiration/
    """
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.now() + func.lifetime
        func.verbose = verbose

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if func.verbose:
                print(f"Current Time: {datetime.now()}, Cache expiration: {func.expiration}")
            if datetime.now() >= func.expiration:
                if func.verbose:
                    print("Cache lifetime expired, retrieving data")
                func.cache_clear()
                func.expiration = datetime.now() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache