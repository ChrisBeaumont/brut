from functools import wraps
import logging
from time import time

logging.getLogger(__name__).setLevel(logging.INFO)

def profile(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        dt = time() - t0
        logging.getLogger(__name__).info("Runtime for %s: %0.2f s" %
                                         (func.__name__, dt))
        return result

    return wrapper
