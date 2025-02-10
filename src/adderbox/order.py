from collections.abc import (
    Callable,
    Iterable,
    Sequence,
)
from typing import Protocol


class Cmp(Protocol):
    """Support less than comparison."""

    def __lt__(self, other, /) -> bool: ...


def lower[C: Cmp](values: Sequence[C], value: C) -> int:
    return lower_by(values, value, key=lambda x: x)


def lower_by[T, C: Cmp](items: Sequence[T], target: C, key: Callable[[T], C]) -> int:
    def find(lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        mid = (lo + hi) // 2  # prefer left on equal
        return find(mid + 1, hi) if key(items[mid]) < target else find(lo, mid)

    return find(0, len(items))


def upper[C: Cmp](values: Sequence[C], value: C) -> int:
    return upper_by(values, value, key=lambda x: x)


def upper_by[T, C: Cmp](items: Sequence[T], target: C, key: Callable[[T], C]) -> int:
    def find(lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        mid = (lo + hi) // 2  # prefer right on equal
        return find(lo, mid) if target < key(items[mid]) else find(mid + 1, hi)

    return find(0, len(items))


def sort[C: Cmp](values: Iterable[C]) -> list[C]:
    return sort_by(values, key=lambda x: x)


def sort_by[T, C: Cmp](items: Iterable[T], key: Callable[[T], C]) -> list[T]:
    def conquer(lo: int, hi: int) -> None:
        if (hi - lo) <= 1:
            return
        end = hi - 1
        pivot = key(buffer[end])
        base: int = lo
        for idx in range(lo, end):
            if pivot < key(buffer[idx]):
                continue
            buffer[base], buffer[idx] = buffer[idx], buffer[base]
            base += 1
        else:
            buffer[base], buffer[end] = buffer[end], buffer[base]

        conquer(lo, base)
        conquer(base + 1, hi)

    buffer = list(items)
    conquer(0, len(buffer))
    return buffer
