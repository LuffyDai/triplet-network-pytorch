from misc import *
import os
from contextlib import contextmanager


@contextmanager
def logging_context(path=None, level=None):
    from logbook import StderrHandler, FileHandler
    from logbook.compat import redirected_logging
    with StderrHandler(level=level or 'INFO').applicationbound():
        if path:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with FileHandler(path, bubble=True).applicationbound():
                with redirected_logging():
                    yield
        else:
            with redirected_logging():
                yield