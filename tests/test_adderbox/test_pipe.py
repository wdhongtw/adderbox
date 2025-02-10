import unittest
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
)

from adderbox import pipe


class TestPiper(unittest.TestCase):

    def test_piper(self) -> None:
        def add_one(val: int) -> int:
            return val + 1

        adder = pipe.Piper(add_one)
        self.assertEqual(2, 1 | adder)


class TestDecorator(unittest.TestCase):

    def test_on_items(self) -> None:
        @pipe.on
        def any(conditions: Iterable[bool]) -> bool:
            for condition in conditions:
                if condition:
                    return True
            return False

        self.assertEqual(False, [] | any())
        self.assertEqual(True, [True, False] | any())
        self.assertEqual(False, [False, False, False] | any())

    def test_curry(self) -> None:

        @pipe._curry_tail
        def filter[_T](items: Iterable[_T], predicate: Callable[[_T], bool]) -> Iterator[_T]:  # fmt: skip
            for item in items:
                if not predicate(item):
                    continue
                yield item

        select: Callable[[int], bool] = lambda val: val % 2 == 0
        results = filter(select)(range(10))
        self.assertEqual([0, 2, 4, 6, 8], list(results))

    def test_pipe(self) -> None:

        @pipe._pipe
        @pipe._curry_tail
        def filter[_T](items: Iterable[_T], predicate: Callable[[_T], bool]) -> Iterator[_T]:  # fmt: skip
            for item in items:
                if not predicate(item):
                    continue
                yield item

        select: Callable[[int], bool] = lambda val: val % 2 == 0
        results = range(10) | filter(select)
        self.assertEqual([0, 2, 4, 6, 8], list(results))


class TestFunction(unittest.TestCase):

    def test_filter(self) -> None:
        items = range(10)
        select: Callable[[int], bool] = lambda val: val % 2 == 0
        results = items | pipe.filter(select)
        self.assertEqual([0, 2, 4, 6, 8], list(results))

    def test_map(self) -> None:
        items = range(5)
        transform: Callable[[int], int] = lambda val: val * 2
        results = items | pipe.map(transform)
        self.assertEqual([0, 2, 4, 6, 8], list(results))

    def test_reduce(self) -> None:
        items = range(5)
        combine: Callable[[int, int], int] = lambda a, b: a + b
        result = items | pipe.reduce(combine, 0)
        self.assertEqual(10, result)
