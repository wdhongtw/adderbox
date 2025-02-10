"""
control is a module to provide some utility to work with context manager style.
"""

from collections.abc import (
    Callable,
    Iterator,
)
from contextlib import (
    AbstractContextManager,
    contextmanager,
)


def build_cm[T](resource: T, close: Callable[[], None]) -> AbstractContextManager[T]:
    """
    Build a context manager from a ready resource and a cleanup function.

    An alternative to contextlib.contextmanager if user do not like
    the style of decorator and nested function.
    """

    @contextmanager
    def wrapped() -> Iterator[T]:
        yield resource
        close()

    return wrapped()


def extract_cm[T](cm: AbstractContextManager[T]) -> tuple[T, Callable[[], None]]:
    """
    Extract context manager into a ready resource and the cleanup function.

    Notice that the __exit__ cleanup will lose the ability to examine exception.
    """

    def close() -> None:
        cm.__exit__(None, None, None)

    resource: T = cm.__enter__()
    return (resource, close)


def on_exit(close: Callable[[], None]) -> AbstractContextManager[None]:
    """
    Pack a cleanup function into a context manager.

    Generalized version of contextlib.closing.
    """
    return build_cm(None, close)
