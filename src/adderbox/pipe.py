"""
pipe is a module that make it easy to write higher-order pipeline function
"""

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
)
from typing import Concatenate


class Piper[T, R]:
    """
    Piper[T, R] is a function that accept T and return R

    call the piper with "value_t | piper_t_r"
    """

    def __init__(self, func: Callable[[T], R]) -> None:
        self._func = func

    def __ror__(self, items: T) -> R:
        return self._func(items)


def on[**P, T, R](
    func: Callable[Concatenate[T, P], R],
) -> Callable[P, Piper[T, R]]:  # fmt: skip
    """
    "on" decorates a func into pipe-style function.

    The result function first takes the arguments, excluding first,
    and returns an object that takes the first argument through "|" operator.
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Piper[T, R]:
        def apply(head: T) -> R:
            return func(head, *args, **kwargs)

        return Piper(apply)

    return wrapped


def _curry_tail[**P, T, R](
    func: Callable[Concatenate[T, P], R],
) -> Callable[P, Callable[[T], R]]:  # fmt: skip
    """
    "curry_tail" decorates a func to take rest arguments first as a HOF.
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Callable[[T], R]:
        def apply(head: T) -> R:
            return func(head, *args, **kwargs)

        return apply

    return wrapped


def _pipe[**P, T, R](
    func: Callable[P, Callable[[T], R]],
) -> Callable[P, Piper[T, R]]:  # fmt: skip
    """
    "pipe" decorates a func to turn the return-value into a Piper
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Piper[T, R]:
        return Piper(func(*args, **kwargs))

    return wrapped


@on
def filter[T](
    items: Iterable[T],
    predicate: Callable[[T], bool],
) -> Iterator[T]:  # fmt: skip
    for item in items:
        if not predicate(item):
            continue
        yield item


@on
def map[T, U](
    items: Iterable[T],
    transform: Callable[[T], U],
) -> Iterator[U]:  # fmt: skip
    for item in items:
        yield transform(item)


@on
def reduce[T, U](
    items: Iterable[T],
    combine: Callable[[U, T], U],
    init: U
) -> U:  # fmt: skip
    value = init
    for item in items:
        value = combine(value, item)
    return value
