from pathlib import Path


def to_path(func):
    def new_func(self, path):
        return func(self, Path(path))

    return new_func
