import numpy as np
from functools import wraps, partial
from time import time
import os, inspect, gc


LOGLEVEL = int(os.getenv("LOGLEVEL", 0))
print = partial(print, flush=True)


def val_from_fchk(key, file_path):
    flag_read = False
    expect_size = -1
    vec = []
    with open(file_path, "r") as file:
        for l in file:
            if l[:len(key)] == key:
                try:
                    expect_size = int(l[len(key):].split()[2])
                    flag_read = True
                    continue
                except IndexError:
                    try:
                        return float(l[len(key):].split()[1])
                    except IndexError:
                        continue
            if flag_read:
                try:
                    vec += [float(i) for i in l.split()]
                except ValueError:
                    break
    if len(vec) != expect_size:
        raise ValueError("Number of expected size is not consistent with read-in size!")
    return np.array(vec)


def timing(f):
    # Answer from https://codereview.stackexchange.com/a/169876
    #             https://stackoverflow.com/a/17065634
    # Should be only using in class functions
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        if LOGLEVEL >= 2:
            stack = inspect.stack()
            the_class = stack[1][0].f_locals["self"].__class__.__qualname__
            the_method = stack[1][0].f_code.co_name
            print(" {0:30s}, {1:50s} Elapsed time: {2:25.10f}".format(f.__qualname__, the_class + "." + the_method + "()", end-start))
        return result
    return wrapper


def gccollect(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        gc.collect()
        return result
    return wrapper
