from pathlib import Path
from typing import Callable, Any, Union


def to_path(func: Callable) -> Callable:
    """
    Decorator to convert a string path argument to a `Path` object.

    :param func: The function to decorate.
    :return: The decorated function.

    .. code-block:: python

        @to_path
        def save_file(self, path: Path):
            path.write_text("Hello, World!")
    """

    def new_func(self: Any, path: Union[str, Path]) -> Any:
        return func(self, Path(path))

    return new_func
