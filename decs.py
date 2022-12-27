"""
Trans Rights are Human Rights

A small collection of function decorators
"""
# SYSTEM IMPORTS
import time

# STANDARD LIBRARY IMPORTS

# LOCAL APPLICATION IMPORTS

def timeit(method):
    def timed(*args, **kw):
        time_start = time.time()
        result = method(*args, **kw)
        time_end = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((time_end - time_start) * 1000)
        else:
            method_time = (time_end - time_start) * 1000
            print(method.__name__, f"{method_time:.2f}")
        return result

    return timed
