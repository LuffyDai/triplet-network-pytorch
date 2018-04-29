import numpy as np
import os

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

from contextlib import contextmanager


@contextmanager
def Context(log=None, parallel=False, level=None):
    from utils import logging_context
    with logging_context(log, level=level):
        if not parallel:
            yield
        else:
            import joblib as jb
            from multiprocessing import cpu_count
            with jb.Parallel(n_jobs=cpu_count()) as par:
                Context.parallel = par
                yield
