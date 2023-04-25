import re
import numpy as np
import time
from functools import wraps

def data_normalization(data: list or np.ndarray):
    """
    Simply Normalization

    Args:
        data(list or np.ndarray)

    Returns:
        normalized data(list or np.ndarray)

    """
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range

def localsToDict(input):
    """
    turn 'ndarray' in locals() to useful dict()
    """
    outDict = dict()
    for key, value in input.items():
        # print(type(value))
        valueTypeName = type(value).__name__

        if re.search('ndarray', valueTypeName):
            outDict[key] = value
        # print(type(outDict))
    return outDict

def timefn(fn):
    """
    A timer and a wrap.
    Using @timefn in front of the function definition for timing the function.
    """
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"{fn.__name__} took {t2 - t1: .5f} s.")
        return result
    return measure_time 