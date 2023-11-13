
from logging import *

from .ansi import ANSI


# https://gist.github.com/after-the-sunrise/751035b06738302ad920adf8bdce0a3f
class LeveledFormatter(Formatter):
    _formats = {}

    def __init__(self, *args, **kwargs):
        super(LeveledFormatter, self).__init__(*args, **kwargs)

    def set_formatter(self, level, formatter):
        self._formats[level] = formatter

    def format(self, record):
        f = self._formats.get(record.levelno)

        if f is None:
            f = super(LeveledFormatter, self)

        return f.format(record)


def add_arguments(parser):
    """
    Add logging arguments to argparse.ArgumentParser
    """


def basic_config(args):
    formatter = LeveledFormatter("[%(module)s] %(message)s")
    formatter.set_formatter(INFO, Formatter(f"{ANSI.set_gray()}[%(module)s] %(message)s{ANSI.reset()}"))
    formatter.set_formatter(WARNING, Formatter(f"{ANSI.set_bright_yellow()}[%(module)s] %(message)s{ANSI.reset()}"))

    handler = StreamHandler()
    handler.setFormatter(formatter)

    basicConfig(level=DEBUG, handlers=[handler])
